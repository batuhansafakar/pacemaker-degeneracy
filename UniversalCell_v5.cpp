#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <random>
#include <chrono>
#include <string>
#include <algorithm>
#include <numeric>
#include <omp.h>

using namespace std;

// =========================================================================
// UNIVERSAL CELL v5.0
// Yenilikler:
//   - Faz bazlı CellPhaseConfig (DT, Ca_vol, E_NCX dinamik, bounds, masks)
//   - E_NCX artık Vm ve Ca'ya göre her adımda hesaplanıyor
//   - Ca_vol hücre tipine göre faz config'inden geliyor
//   - Kompartman ağırlıkları (soma/dendrit split) üretim aşamasında
//   - Firing mode sınıflandırması: spontaneous / evokable / silent
//   - Evoke testi: freq=0 hücrelere 200ms depolarize pulse uygulanır
//   - Faz bazlı validasyon sınırları (freq, CV, V_max, V_min)
//   - Her satıra phase_id, compartment_mode, firing_mode yazılır
//
// Derle: g++ -O3 -march=native -std=c++17 -fopenmp UniversalCell_v5.cpp -o uc5.exe
// Calistir: uc5.exe 1500000
// =========================================================================

// -------------------------------------------------------------------------
// GENEL SABİTLER
// -------------------------------------------------------------------------
static constexpr float T_MAX        = 12000.0f;   // ms
static constexpr float WARMUP_MS    = 2000.0f;
static constexpr float RECORD_SEC   = (T_MAX - WARMUP_MS) / 1000.0f;
static constexpr float CM           = 1.0f;        // µF/cm²

// Reversal potansiyeller — Na ve Ca dinamik hesaplanacak
static constexpr float ECa_GLOBAL   = 120.0f;   // Ca_o sabit (2 mM extrasell.)
static constexpr float EK_GLOBAL    = -90.0f;
static constexpr float Eh_GLOBAL    = -35.0f;
static constexpr float ELeak_GLOBAL = -70.0f;

// Na dinamiği sabitleri
static constexpr float NAI_REST     = 10.0f;    // mM — dinlenme Na_i
static constexpr float NAO          = 145.0f;   // mM — extrasellüler Na (sabit)
static constexpr float CAO          = 2.0f;     // mM — extrasellüler Ca (sabit)
static constexpr float RT_F         = 25.7f;    // mV (37°C) — RT/F
// NaK-ATPase: elektrojenik pompa (3Na dışarı, 2K içeri)
// J_NaKATPase = Imax * (Na_i^1.5 / (Na_i^1.5 + KM_Na^1.5))
static constexpr float NAKATPASE_IMAX = 0.08f;  // mM/ms — maksimum pompa hızı
static constexpr float NAKATPASE_KM   = 10.0f;  // mM — Na_i yarı doyum sabiti
// Na_i → Vm dönüşüm katsayısı (Ca_vol ile paralel, nöron geometrisi)
// Na akımları mA/cm², Na_i mM, dönüşüm: F*Vol/Area ≈ 1/(9.65e4 * ca_vol_na)
// Tipik değer: ~1e-4 mM·cm²/(mA·ms) — faz config'den gelecek

// Evoke test parametreleri
static constexpr float EVOKE_CURRENT    = 2.0f;   // µA/cm² depolarize pulse
static constexpr float EVOKE_START_MS   = 500.0f;
static constexpr float EVOKE_DUR_MS     = 200.0f;
static constexpr float EVOKE_T_MAX      = 2000.0f; // ms — kısa test
static constexpr float EVOKE_THRESH_HZ  = 0.5f;   // bu frekansın üstü = evokable

static constexpr float SENTINEL_FREQ = -1.0f;
static constexpr float SENTINEL_V    = -999.0f;

// -------------------------------------------------------------------------
// KANAL İNDEKSLERİ (g vektörü sırası)
// -------------------------------------------------------------------------
// 0:NaF  1:NaP  2:NaR  3:HCN  4:Kv1  5:Kv2  6:Kv3  7:Kv4
// 8:KCNQ 9:Kir  10:CaT 11:CaL 12:CaPQ 13:CaN 14:CaR
// 15:SK  16:BK  17:Leak 18:PMCA 19:SERCA 20:NCX
static constexpr int N_CHANNELS = 21;

// -------------------------------------------------------------------------
// KOMPARTMAN AĞIRLIKLARI
// Soma fraksiyonu — dendrit = 1 - soma
// Sıra: NaF,NaP,NaR,HCN,Kv1,Kv2,Kv3,Kv4,KCNQ,Kir,
//       CaT,CaL,CaPQ,CaN,CaR,SK,BK,Leak,PMCA,SERCA,NCX
// -------------------------------------------------------------------------
static const float SINGLE_COMP[N_CHANNELS] = {
    1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1,1
};
// Pace-maker (SNc, VTA, LC, DR, SCN): HCN ve CaT dendrite ağır
static const float PACEMAKER_SOMA[N_CHANNELS] = {
    0.95f,0.90f,0.90f,0.20f,0.80f,0.80f,0.95f,0.15f,0.80f,0.30f,
    0.30f,0.60f,0.25f,0.25f,0.25f,0.40f,0.40f,0.77f,1.0f,1.0f,1.0f
};
// Burst/osillatör (STN, GPe, Purkinje, IO, TC): CaL ve CaPQ dendrite
static const float BURST_SOMA[N_CHANNELS] = {
    0.90f,0.85f,0.85f,0.30f,0.75f,0.75f,0.90f,0.20f,0.75f,0.40f,
    0.25f,0.35f,0.20f,0.20f,0.20f,0.35f,0.35f,0.70f,1.0f,1.0f,1.0f
};
// İnhibitör (SNr, GPi, striatal ChAT): soma-ağır, dendrit minimal
static const float INHIBIT_SOMA[N_CHANNELS] = {
    0.98f,0.95f,0.95f,0.50f,0.90f,0.90f,0.98f,0.50f,0.90f,0.60f,
    0.50f,0.70f,0.50f,0.50f,0.50f,0.60f,0.60f,0.85f,1.0f,1.0f,1.0f
};

// -------------------------------------------------------------------------
// VALIDASYON KRİTERLERİ (faz bazlı)
// -------------------------------------------------------------------------
struct ValidationCriteria {
    float freq_min, freq_max;   // Hz — spontan ateşleme penceresi
    float cv_max;               // ISI CV üst sınırı
    float vmax_min;             // spike tepe alt sınırı (mV)
    float vmin_max;             // trough üst sınırı (mV)
    float sentinel_max_ratio;   // kabul edilebilir sentinel oranı
};

// -------------------------------------------------------------------------
// FAZ KONFİGÜRASYONU
// -------------------------------------------------------------------------
struct CellPhaseConfig {
    string phase_id;
    string compartment_mode;    // "single" | "dual"
    const float* soma_weights;  // kompartman ağırlıkları

    float dt;
    float ca_vol;               // ICa → Ca dönüşüm katsayısı
    float na_vol;               // INa → Na_i dönüşüm katsayısı (ca_vol ile orantılı)
    float ca_er_init;
    float km_serca;
    float km_ryr;
    float k_ryr;
    float sk_kd;
    float bk_kd;
    float gc;                   // axial coupling (dual mod için)

    float mask_thresh;          // kanal sıfır olasılığı eşiği
    int   n_sim;

    ValidationCriteria valid;

    vector<pair<float,float>> bounds; // her kanal için [min, max]
};

// -------------------------------------------------------------------------
// FAZ TANIMLARI
// -------------------------------------------------------------------------
static CellPhaseConfig make_phase_A() {
    // Pace-maker: SNc, VTA, LC, Dorsal Raphe, SCN
    CellPhaseConfig c;
    c.phase_id         = "A_pacemaker";
    c.compartment_mode = "dual";
    c.soma_weights     = PACEMAKER_SOMA;
    c.dt               = 0.001f;
    c.ca_vol           = 4e-5f;   // küçük soma ~10µm çap
    c.na_vol           = 8e-5f;   // Na_i: ca_vol*2 (monovalent, yüksek turnover)
    c.ca_er_init       = 100.0f;
    c.km_serca         = 0.03f;
    c.km_ryr           = 0.02f;
    c.k_ryr            = 0.01f;
    c.sk_kd            = 0.015f;  // nanodomain
    c.bk_kd            = 0.05f;
    c.gc               = 0.35f;
    c.mask_thresh      = 0.40f;
    c.valid            = {1.0f, 15.0f, 0.15f, 20.0f, -60.0f, 0.03f};
    c.bounds = {
        {40,180},  // NaF
        {0,1.2f},  // NaP — pace-maker'da düşük
        {0,6},     // NaR
        {0.5f,5},  // HCN — zorunlu pace-maker komponenti
        {0,30},    // Kv1
        {0,30},    // Kv2
        {10,150},  // Kv3
        {0,30},    // Kv4
        {0,8},     // KCNQ
        {0,4},     // Kir
        {0.1f,2.5f},{0,3},{0,2},{0,2},{0,2}, // Ca kanalları — CaT zorunlu
        {0,8},{0,12},  // SK, BK
        {0.02f,0.35f}, // Leak
        {0.01f,0.15f},{0.01f,0.15f},{0,3}   // PMCA, SERCA, NCX
    };
    return c;
}

static CellPhaseConfig make_phase_B() {
    // Burst/osillatör: STN, GPe, Purkinje, Inferior Olive, TC thalamus
    CellPhaseConfig c;
    c.phase_id         = "B_burst";
    c.compartment_mode = "dual";
    c.soma_weights     = BURST_SOMA;
    c.dt               = 0.001f;
    c.ca_vol           = 1.5e-4f; // büyük soma ~20µm çap
    c.na_vol           = 3e-4f;   // büyük soma Na_i katsayısı
    c.ca_er_init       = 200.0f;  // dolu ER — burst için kritik
    c.km_serca         = 0.03f;
    c.km_ryr           = 0.02f;
    c.k_ryr            = 0.015f;
    c.sk_kd            = 0.015f;
    c.bk_kd            = 0.05f;
    c.gc               = 0.50f;   // güçlü axial coupling
    c.mask_thresh      = 0.35f;   // daha az sıfır — kanal kombinasyonları kritik
    c.valid            = {5.0f, 120.0f, 0.40f, 15.0f, -65.0f, 0.05f};
    c.bounds = {
        {40,220},  // NaF
        {0,1.5f},  // NaP
        {0,8},     // NaR
        {0,4},     // HCN
        {0,40},{0,40}, // Kv1, Kv2
        {15,280},  // Kv3 — burst hücrelerinde yüksek
        {0,40},    // Kv4
        {0,8},     // KCNQ
        {0,4},     // Kir
        {0,2.5f},{0.1f,4},{0,2.5f},{0,2.5f},{0,2.5f}, // Ca — CaL zorunlu
        {0,8},{0,15},  // SK, BK
        {0.02f,0.40f}, // Leak
        {0,0.18f},{0,0.18f},{0,4}   // PMCA, SERCA, NCX
    };
    return c;
}

static CellPhaseConfig make_phase_C() {
    // İnhibitör: SNr, GPi, Striatal ChAT
    CellPhaseConfig c;
    c.phase_id         = "C_inhibitory";
    c.compartment_mode = "single";
    c.soma_weights     = INHIBIT_SOMA;  // single'da kullanılmaz
    c.dt               = 0.001f;
    c.ca_vol           = 7e-5f;   // orta boy soma
    c.na_vol           = 1.4e-4f; // orta soma Na_i katsayısı
    c.ca_er_init       = 100.0f;
    c.km_serca         = 0.03f;
    c.km_ryr           = 0.02f;
    c.k_ryr            = 0.01f;
    c.sk_kd            = 0.015f;
    c.bk_kd            = 0.05f;
    c.gc               = 0.0f;
    c.mask_thresh      = 0.50f;
    c.valid            = {10.0f, 100.0f, 0.20f, 25.0f, -58.0f, 0.04f};
    c.bounds = {
        {60,350},  // NaF — inhibitör hücreler yüksek NaF
        {0,1.0f},  // NaP
        {0,8},     // NaR
        {0,3},     // HCN
        {0,40},{0,40}, // Kv1, Kv2
        {20,400},  // Kv3 — inhibitör'ün ayırt edici özelliği
        {0,30},    // Kv4
        {0,8},     // KCNQ
        {0,5},     // Kir
        {0,2},{0,3},{0,2},{0,2},{0,2}, // Ca
        {0,6},{0,10},  // SK, BK
        {0.02f,0.45f}, // Leak
        {0,0.15f},{0,0.15f},{0,3}   // PMCA, SERCA, NCX
    };
    return c;
}

static CellPhaseConfig make_phase_D() {
    // Genel keşif — geniş bounds, tek kompartman
    CellPhaseConfig c;
    c.phase_id         = "D_general";
    c.compartment_mode = "single";
    c.soma_weights     = SINGLE_COMP;
    c.dt               = 0.001f;
    c.ca_vol           = 5e-5f;
    c.na_vol           = 1e-4f;   // genel faz Na_i katsayısı
    c.ca_er_init       = 100.0f;
    c.km_serca         = 0.03f;
    c.km_ryr           = 0.02f;
    c.k_ryr            = 0.01f;
    c.sk_kd            = 0.015f;
    c.bk_kd            = 0.05f;
    c.gc               = 0.0f;
    c.mask_thresh      = 0.50f;
    c.valid            = {0.2f, 250.0f, 0.80f, 0.0f, -80.0f, 0.08f};
    c.bounds = {
        {20,250},{0,2},{0,10},{0,5},
        {0,50},{0,50},{10,300},{0,50},
        {0,10},{0,5},
        {0,3},{0,5},{0,3},{0,3},{0,3},
        {0,10},{0,20},
        {0.01f,0.5f},
        {0,0.2f},{0,0.2f},{0,5}
    };
    return c;
}

// -------------------------------------------------------------------------
// YARDIMCI FONKSİYONLAR
// -------------------------------------------------------------------------
inline float x_inf(float V, float Vh, float k) {
    float z = (V - Vh) / k;
    if (z < -60.0f) z = -60.0f;
    if (z >  60.0f) z =  60.0f;
    return 1.0f / (1.0f + expf(-z));
}

inline void step_gate(float& x, float V, float Vh, float k, float tau, float dt) {
    x += (dt / tau) * (x_inf(V, Vh, k) - x);
}

// Dinamik ENa — Na_i Nernst
// Dinlenme (Na_i=10): ENa = 25.7*ln(14.5) ≈ 68 mV
// Yükleme (Na_i=15):  ENa = 25.7*ln(9.67) ≈ 59 mV → efektif depolarizasyon azalır
inline float compute_ena(float Na_i) {
    if(Na_i < 1.0f) Na_i = 1.0f;
    return RT_F * logf(NAO / Na_i);
}

// Dinamik E_NCX — hem Na_i hem Ca_i değişiyor
// E_NCX = 3*ENa(Na_i) - 2*ECa(Ca_i)
// Dinlenme: 3*68 - 2*120 ≈ -36 mV (düşük Ca → NCX Ca giriş modu yakın)
// Spike Ca yükselince: ECa düşer, E_NCX pozitife kayar → NCX Ca çıkarmaya geçer ✓
// Na_i yüklenince:     ENa düşer, E_NCX negatife kayar → NCX kapasitesi azalır ✓
inline float compute_encx(float Na_i, float Ca_in) {
    if(Na_i < 1.0f)     Na_i     = 1.0f;
    float Ca_i_mM = Ca_in * 1e-3f;          // µM → mM
    if(Ca_i_mM < 1e-6f) Ca_i_mM = 1e-6f;
    float ENa_dyn = RT_F * logf(NAO / Na_i);
    float ECa_dyn = (RT_F / 2.0f) * logf(CAO / Ca_i_mM);
    return 3.0f * ENa_dyn - 2.0f * ECa_dyn;
}

// -------------------------------------------------------------------------
// ANA SİMÜLASYON — config ile
// -------------------------------------------------------------------------
struct CellResult {
    vector<float> g;
    float freq, isi_cv, v_max, v_min;
    float na_i_final;
    string firing_mode;
};

CellResult simulate_cell(const vector<float>& g, const CellPhaseConfig& cfg,
                         bool do_evoke_test = false) {

    const float dt        = cfg.dt;
    const float ca_vol    = cfg.ca_vol;
    const float na_vol    = cfg.na_vol;
    const float sk_kd     = cfg.sk_kd;
    const float bk_kd     = cfg.bk_kd;
    const float km_serca  = cfg.km_serca;
    const float km_ryr    = cfg.km_ryr;
    const float k_ryr     = cfg.k_ryr;
    const float ca_er_init = cfg.ca_er_init;

    const float sk_kd4 = sk_kd * sk_kd * sk_kd * sk_kd;

    float t_max  = do_evoke_test ? EVOKE_T_MAX : T_MAX;
    float warmup = do_evoke_test ? 100.0f       : WARMUP_MS;
    int n_steps  = (int)(t_max / dt);
    int warmup_s = (int)(warmup / dt);
    float rec_sec = do_evoke_test ? (EVOKE_T_MAX - 100.0f) / 1000.0f
                                  : RECORD_SEC;

    float Vm    = -65.0f;
    float Ca_in = 0.0f;
    float Ca_ER = ca_er_init;
    float Na_i  = NAI_REST;     // mM — dinlenme değerinden başla

    float mNa=x_inf(Vm,-40,6),   hNa=x_inf(Vm,-65,-6);
    float mNaP=x_inf(Vm,-55,6);
    float mNaR=x_inf(Vm,-30,5),  hNaR=x_inf(Vm,-30,-5);
    float mHCN=x_inf(Vm,-75,-8);
    float nKv1=x_inf(Vm,-35,7),  nKv2=x_inf(Vm,-10,9);
    float nKv3=x_inf(Vm,-5,8);
    float aKv4=x_inf(Vm,-50,7),  bKv4=x_inf(Vm,-70,-6);
    float mKCNQ=x_inf(Vm,-40,6);
    float bCaT=x_inf(Vm,-70,-5);
    float dCaL=x_inf(Vm,-20,6),  fCaL=x_inf(Vm,-35,-7);
    float dCaPQ=x_inf(Vm,-10,6), fCaPQ=x_inf(Vm,-30,-6);
    float dCaN=x_inf(Vm,-15,6),  fCaN=x_inf(Vm,-35,-6.5f);
    float dCaR=x_inf(Vm,-25,6),  fCaR=x_inf(Vm,-45,-7);
    float mSK=0.0f, mBK=x_inf(Vm,-25,7);

    float v_max=-200.0f, v_min=200.0f;
    vector<float> spike_times; spike_times.reserve(500);
    bool in_spike=false;

    int evoke_start = (int)(EVOKE_START_MS / dt);
    int evoke_end   = (int)((EVOKE_START_MS + EVOKE_DUR_MS) / dt);

    for (int i = 0; i < n_steps; ++i) {

        // Gate güncellemeleri
        if(g[0]>0){step_gate(mNa,Vm,-40,6,0.2f,dt); step_gate(hNa,Vm,-65,-6,1.2f,dt);}
        if(g[1]>0) step_gate(mNaP,Vm,-55,6,6.0f,dt);
        if(g[2]>0){step_gate(mNaR,Vm,-30,5,1.5f,dt); step_gate(hNaR,Vm,-30,-5,0.3f,dt);}
        if(g[3]>0) step_gate(mHCN,Vm,-75,-8,200.0f,dt);
        if(g[4]>0) step_gate(nKv1,Vm,-35,7,6.0f,dt);
        if(g[5]>0) step_gate(nKv2,Vm,-10,9,15.0f,dt);
        if(g[6]>0) step_gate(nKv3,Vm,-5,8,2.0f,dt);
        if(g[7]>0){step_gate(aKv4,Vm,-50,7,12.0f,dt); step_gate(bKv4,Vm,-70,-6,30.0f,dt);}
        if(g[8]>0) step_gate(mKCNQ,Vm,-40,6,80.0f,dt);
        if(g[10]>0) step_gate(bCaT,Vm,-70,-5,40.0f,dt);
        if(g[11]>0){step_gate(dCaL,Vm,-20,6,3.0f,dt);  step_gate(fCaL,Vm,-35,-7,120.0f,dt);}
        if(g[12]>0){step_gate(dCaPQ,Vm,-10,6,2.5f,dt); step_gate(fCaPQ,Vm,-30,-6,60.0f,dt);}
        if(g[13]>0){step_gate(dCaN,Vm,-15,6,3.5f,dt);  step_gate(fCaN,Vm,-35,-6.5f,70.0f,dt);}
        if(g[14]>0){step_gate(dCaR,Vm,-25,6,4.0f,dt);  step_gate(fCaR,Vm,-45,-7,90.0f,dt);}

        // SK — Hill kooperatif bağlanma
        float Ca4 = Ca_in*Ca_in*Ca_in*Ca_in;
        if(g[15]>0) mSK += (dt/25.0f)*(Ca4/(Ca4+sk_kd4)-mSK);

        // BK — Ca'ya bağlı V_half kayması
        float vh_bk = 15.0f - 50.0f*(Ca_in/(Ca_in+bk_kd));
        if(g[16]>0) step_gate(mBK,Vm,vh_bk,7.0f,2.0f,dt);

        // Akım hesapları — ENa dinamik (Na_i Nernst)
        float ENa_dyn = compute_ena(Na_i);
        float INaF  = g[0]*(mNa*mNa*mNa)*hNa*(Vm-ENa_dyn);
        float INaP  = g[1]*mNaP*(Vm-ENa_dyn);
        float INaR  = g[2]*mNaR*hNaR*(Vm-ENa_dyn);
        float INa_tot = INaF + INaP + INaR;   // toplam Na girişi
        float Ih    = g[3]*mHCN*(Vm-Eh_GLOBAL);
        float IKv1  = g[4]*(nKv1*nKv1)*(Vm-EK_GLOBAL);
        float IKv2  = g[5]*(nKv2*nKv2)*(Vm-EK_GLOBAL);
        float IKv3  = g[6]*(nKv3*nKv3*nKv3*nKv3)*(Vm-EK_GLOBAL);
        float IKv4  = g[7]*(aKv4*aKv4*aKv4)*bKv4*(Vm-EK_GLOBAL);
        float IKCNQ = g[8]*mKCNQ*(Vm-EK_GLOBAL);
        float IKir  = g[9]*x_inf(Vm,-80,-6)*(Vm-EK_GLOBAL);

        float aCaT  = x_inf(Vm,-60,5.8f);
        float ICaT  = g[10]*(aCaT*aCaT*aCaT)*bCaT*(Vm-ECa_GLOBAL);
        float ICaL  = g[11]*(dCaL*dCaL)*fCaL*(Vm-ECa_GLOBAL);
        float ICaPQ = g[12]*(dCaPQ*dCaPQ)*fCaPQ*(Vm-ECa_GLOBAL);
        float ICaN  = g[13]*(dCaN*dCaN)*fCaN*(Vm-ECa_GLOBAL);
        float ICaR  = g[14]*(dCaR*dCaR)*fCaR*(Vm-ECa_GLOBAL);
        float ICa_tot = ICaT+ICaL+ICaPQ+ICaN+ICaR;

        float ISK   = g[15]*mSK*(Vm-EK_GLOBAL);
        float IBK   = g[16]*mBK*(Vm-EK_GLOBAL);
        float ILeak = g[17]*(Vm-ELeak_GLOBAL);

        // Dinamik E_NCX — hem Na_i hem Ca_i ile
        float E_NCX = compute_encx(Na_i, Ca_in);
        // INCX işaret kuralı:
        //   Vm < E_NCX → INCX negatif (inward) → net +1 yük içeri → 3Na girer, 1Ca çıkar → Forward mode
        //   Vm > E_NCX → INCX pozitif (outward) → net +1 yük dışarı → 3Na çıkar, 1Ca girer → Reverse mode
        float INCX  = g[20]*(Ca_in/(Ca_in+1.0f))*(Vm-E_NCX);

        // Ca fluks: Forward (INCX<0) → Ca dışarı → Ca_in azalır → +INCX işareti
        //           Reverse (INCX>0) → Ca içeri → Ca_in artar → +INCX işareti
        // Yani Ca_in değişimi INCX ile AYNI işaretli olmalı
        float J_NCX_Ca = ca_vol * INCX;   // Ca_in üzerindeki etki

        // Na fluks: Forward (INCX<0) → 3Na içeri → Na_i artar → -3*INCX işareti
        //           Reverse (INCX>0) → 3Na dışarı → Na_i azalır → -3*INCX işareti
        // Yani Na_i değişimi -3*INCX ile orantılı
        float J_NCX_Na = -3.0f * na_vol * INCX;  // Na_i üzerindeki etki

        // Ca pompaları ve ER dinamikleri
        float Ca_in2  = Ca_in*Ca_in;
        float KM2s    = km_serca*km_serca;
        float J_SERCA = g[19]*(Ca_in2/(Ca_in2+KM2s));
        float J_PMCA  = g[18]*(Ca_in/(Ca_in+0.2f));
        float J_RyR   = k_ryr*Ca_ER*(Ca_in2/(Ca_in2+km_ryr*km_ryr));

        // Ca_in türevi:
        //   ICa_tot inward (negatif) → Ca girer → -ca_vol*ICa_tot pozitif ✓
        //   PMCA/SERCA → Ca çıkarır → eksi ✓
        //   RyR → ER'den Ca salar → artı ✓
        //   J_NCX_Ca → INCX işaretiyle aynı yönde ✓
        Ca_in += dt*(ca_vol*(-ICa_tot) + J_NCX_Ca - J_PMCA - J_SERCA + J_RyR);
        if(Ca_in < 0.0f) Ca_in = 0.0f;

        Ca_ER += dt*(J_SERCA*50.0f - J_RyR*50.0f);
        if(Ca_ER < 0.0f)    Ca_ER = 0.0f;
        if(Ca_ER > 5000.0f) Ca_ER = 5000.0f;

        // Na_i türevi:
        //   INa_tot inward (negatif) → Na girer → -na_vol*INa_tot pozitif ✓
        //   NaK-ATPase → 3Na dışarı atar → eksi ✓
        //   J_NCX_Na → -3*INCX yönünde ✓
        float Na15  = powf(Na_i, 1.5f);
        float KM15  = powf(NAKATPASE_KM, 1.5f);
        float J_NaK = NAKATPASE_IMAX * (Na15 / (Na15 + KM15));
        Na_i += dt * (na_vol*(-INa_tot) - J_NaK + J_NCX_Na);
        if(Na_i < 1.0f)  Na_i = 1.0f;
        if(Na_i > 50.0f) Na_i = 50.0f;

        // Evoke pulse (sadece evoke test modunda)
        float I_ext = 0.0f;
        if(do_evoke_test && i >= evoke_start && i < evoke_end)
            I_ext = EVOKE_CURRENT;

        float Itot = INaF+INaP+INaR+Ih+IKv1+IKv2+IKv3+IKv4+IKCNQ+IKir
                    +ICa_tot+ISK+IBK+ILeak+INCX;
        Vm += (dt/CM)*(-Itot + I_ext);

        if(isnan(Vm)||Vm>200.0f||Vm<-200.0f)
            return {g, SENTINEL_FREQ, SENTINEL_V, SENTINEL_V, SENTINEL_V, SENTINEL_V, "sentinel"};

        if(i > warmup_s){
            if(Vm > v_max) v_max = Vm;
            if(Vm < v_min) v_min = Vm;
            if(Vm > 0.0f && !in_spike){
                spike_times.push_back(i*dt);
                in_spike = true;
            } else if(Vm < -10.0f && in_spike){
                in_spike = false;
            }
        }
    }

    int sc = (int)spike_times.size();
    float freq = (sc >= 2) ? (sc / rec_sec) : 0.0f;

    float isi_cv = numeric_limits<float>::quiet_NaN();
    if(sc >= 3){
        vector<float> isi; isi.reserve(sc-1);
        for(int j=1;j<sc;++j) isi.push_back(spike_times[j]-spike_times[j-1]);
        float m=0; for(float v:isi) m+=v; m/=isi.size();
        float var=0; for(float v:isi) var+=(v-m)*(v-m); var/=isi.size();
        isi_cv = (m>1e-6f) ? sqrtf(var)/m : 0.0f;
    }

    return {g, freq, isi_cv, v_max, v_min, Na_i, ""};
}

// -------------------------------------------------------------------------
// EVOKE TESTİ — firing_mode belirleme
// freq=0 hücrelerin evokable olup olmadığını test eder
// Ek olarak V_max/V_min hareket kriterini de kontrol eder
// -------------------------------------------------------------------------
string classify_firing_mode(const CellResult& r, const CellPhaseConfig& cfg) {
    if(r.freq > 0.0f) return "spontaneous";

    // V_max/V_min hareketine bak — membran potansiyeli hiç kıpırdamış mı?
    // Hareket etmişse ama eşiği aşamamışsa → evoke adayı
    bool membrane_moved = (r.v_max > -50.0f) && (r.v_min < -55.0f);

    if(!membrane_moved) {
        // Tamamen hareketsiz — gerçek sessiz (inhibisyon altında / down-state)
        return "silent";
    }

    // Membran hareket etmiş ama ateşlememiş — evoke testi yap
    CellResult evoke_result = simulate_cell(r.g, cfg, /*do_evoke_test=*/true);

    if(evoke_result.freq >= EVOKE_THRESH_HZ)
        return "evokable";
    else
        return "silent";
}

// -------------------------------------------------------------------------
// KOMPARTMAN AĞIRLIKLARI UYGULAMA
// Dual mod: g vektörünü soma fraksiyonuyla çarp
// CSV'ye soma değerleri yazılır; downstream dendrit = g * (1-soma_frac)
// -------------------------------------------------------------------------
vector<float> apply_compartment_weights(const vector<float>& g,
                                         const CellPhaseConfig& cfg) {
    if(cfg.compartment_mode == "single") return g;
    vector<float> g_soma(N_CHANNELS);
    for(int c=0;c<N_CHANNELS;++c)
        g_soma[c] = g[c] * cfg.soma_weights[c];
    return g_soma;
}

// -------------------------------------------------------------------------
// FAZ ÇALIŞTIRICI
// -------------------------------------------------------------------------
void run_phase(const CellPhaseConfig& cfg, const string& fname,
               int& total_sentinel, int& total_written) {

    int n_threads = omp_get_max_threads();
    printf("[%s] %d sim / %d thread / dt=%.3f / ca_vol=%.2e\n",
           cfg.phase_id.c_str(), cfg.n_sim, n_threads, cfg.dt, cfg.ca_vol);
    fflush(stdout);

    auto t0 = chrono::high_resolution_clock::now();
    int done=0, sentinel_count=0, last_report=0;

    ofstream out_file(fname, ios::app);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        mt19937 gen(42u ^ (unsigned)(tid * 2654435761u) ^ (unsigned)cfg.phase_id[0]);
        uniform_real_distribution<float> mask_dist(0,1);

        vector<uniform_real_distribution<float>> val_dists;
        val_dists.reserve(N_CHANNELS);
        for(int c=0;c<N_CHANNELS;++c)
            val_dists.emplace_back(cfg.bounds[c].first, cfg.bounds[c].second);

        const int BLOCK = 10;
        vector<string> buf; buf.reserve(BLOCK);
        int local_sentinel = 0;

        #pragma omp for schedule(dynamic, 10) nowait
        for(int i=0; i<cfg.n_sim; ++i) {
            vector<float> g(N_CHANNELS, 0.0f);
            for(int c=0;c<N_CHANNELS;++c){
                // g_Leak (17) her zaman dolu
                if(c==17 || mask_dist(gen) > cfg.mask_thresh)
                    g[c] = val_dists[c](gen);
            }

            // NaF yoksa simülasyonsuz — freq=0 olarak kaydet
            CellResult r;
            if(g[0] < 1.0f){
                r = {g, 0.0f, numeric_limits<float>::quiet_NaN(), -65.0f, -65.0f, NAI_REST, "silent"};
            } else {
                r = simulate_cell(g, cfg, false);
                if(r.freq == SENTINEL_FREQ){ local_sentinel++; continue; }

                // Firing mode sınıflandırması
                r.firing_mode = classify_firing_mode(r, cfg);

                // Kompartman ağırlıkları uygula (dual modda g_soma yazılır)
                r.g = apply_compartment_weights(r.g, cfg);
            }

            // Satır yaz
            string row; row.reserve(320);
            char tmp[32];
            for(float gv : r.g){ snprintf(tmp,sizeof(tmp),"%.4f,",gv); row+=tmp; }
            snprintf(tmp,sizeof(tmp),"%.3f,",r.freq); row+=tmp;
            if(isnan(r.isi_cv)) row+=",";
            else { snprintf(tmp,sizeof(tmp),"%.4f,",r.isi_cv); row+=tmp; }
            snprintf(tmp,sizeof(tmp),"%.3f,%.3f,",r.v_max,r.v_min); row+=tmp;
            snprintf(tmp,sizeof(tmp),"%.2f,",r.na_i_final); row+=tmp;
            row += cfg.phase_id + "," + cfg.compartment_mode + "," + r.firing_mode + "\n";
            buf.push_back(row);

            if((int)buf.size() >= BLOCK){
                #pragma omp critical
                {
                    for(auto& s:buf) out_file<<s;
                    out_file.flush();
                    done += (int)buf.size();
                    sentinel_count += local_sentinel;
                    local_sentinel = 0;

                    if(done - last_report >= 2000){
                        last_report = done;
                        double el = chrono::duration<double>(
                            chrono::high_resolution_clock::now()-t0).count();
                        double rate = done/(el+1e-9);
                        double eta  = (cfg.n_sim-done)/(rate+1e-9);
                        float s_ratio = (done+sentinel_count)>0 ?
                            100.f*sentinel_count/(done+sentinel_count) : 0;
                        printf("  [%s] %d/%d  %%%.0f sim/sn | sentinel:%%%.1f | ETA:%.0f dk\n",
                            cfg.phase_id.c_str(), done, cfg.n_sim,
                            rate, s_ratio, eta/60.0);
                        fflush(stdout);
                    }
                }
                buf.clear();
            }
        }

        if(!buf.empty()){
            #pragma omp critical
            {
                for(auto& s:buf) out_file<<s;
                out_file.flush();
                done += (int)buf.size();
                sentinel_count += local_sentinel;
            }
            buf.clear();
        }
    }

    out_file.close();
    total_written  += done;
    total_sentinel += sentinel_count;

    double el = chrono::duration<double>(chrono::high_resolution_clock::now()-t0).count();
    float s_ratio = (done+sentinel_count)>0 ?
        100.f*sentinel_count/(done+sentinel_count) : 0;
    printf("[%s] Bitti: %d yazildi | sentinel:%%%.1f | %.1f dk | %.0f sim/sn\n\n",
           cfg.phase_id.c_str(), done, s_ratio, el/60.0, done/(el+1e-9));
}

// -------------------------------------------------------------------------
// MAIN
// -------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    int total = (argc>1) ? atoi(argv[1]) : 1500000;

    // Faz dağılımı
    // A: pace-maker   — 30%
    // B: burst        — 25%
    // C: inhibitory   — 20%
    // D: general      — 25%
    CellPhaseConfig phA = make_phase_A(); phA.n_sim = (int)(total * 0.30f);
    CellPhaseConfig phB = make_phase_B(); phB.n_sim = (int)(total * 0.25f);
    CellPhaseConfig phC = make_phase_C(); phC.n_sim = (int)(total * 0.20f);
    CellPhaseConfig phD = make_phase_D(); phD.n_sim = total - phA.n_sim - phB.n_sim - phC.n_sim;

    const string fname = "UniversalCell_v5.csv";

    // Header
    {
        ofstream f(fname);
        // g sütunları (soma değerleri — dual modda soma fraksiyonu)
        f << "g_NaF,g_NaP,g_NaR,g_HCN,g_Kv1,g_Kv2,g_Kv3,g_Kv4,g_KCNQ,g_Kir,"
          << "g_CaT,g_CaL,g_CaPQ,g_CaN,g_CaR,g_SK,g_BK,g_Leak,"
          << "g_PMCA,g_SERCA,g_NCX,"
          << "Freq_Hz,ISI_CV,V_max,V_min,Na_i_final,"
          << "phase_id,compartment_mode,firing_mode\n";
    }

    printf("============================================================\n");
    printf("  UNIVERSAL CELL v5.0\n");
    printf("  Toplam: %d | Thread: %d\n", total, omp_get_max_threads());
    printf("  Faz A (pacemaker): %d | Faz B (burst): %d\n", phA.n_sim, phB.n_sim);
    printf("  Faz C (inhibitor): %d | Faz D (genel):  %d\n", phC.n_sim, phD.n_sim);
    printf("  Yenilikler: dinamik E_NCX | Ca_vol faz-bazli\n");
    printf("              firing_mode: spontaneous/evokable/silent\n");
    printf("              kompartman agirliklari uretim aninda\n");
    printf("  Cikti: %s\n", fname.c_str());
    printf("============================================================\n\n");

    int total_written=0, total_sentinel=0;

    run_phase(phA, fname, total_sentinel, total_written);
    run_phase(phB, fname, total_sentinel, total_written);
    run_phase(phC, fname, total_sentinel, total_written);
    run_phase(phD, fname, total_sentinel, total_written);

    float s_ratio = (total_written+total_sentinel)>0 ?
        100.f*total_sentinel/(total_written+total_sentinel) : 0;

    printf("============================================================\n");
    printf("  TAMAMLANDI\n");
    printf("  Yazilan  : %d satir\n", total_written);
    printf("  Sentinel : %%%.1f\n", s_ratio);
    printf("  Cikti    : %s\n", fname.c_str());
    printf("============================================================\n");
    return 0;
}