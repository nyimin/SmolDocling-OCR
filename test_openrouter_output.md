---
document:
  source_file: "Optimal_Sizing_of_a_Wind_PV_Grid_Connected_Hybrid_System_for_Base_Load_Helsinki_Case.pdf"
  pages: 7
  extraction_method: "OpenRouter/Gemini 2.0 Flash Lite"
  extraction_date: "2026-02-09T02:58:12.777878"
  confidence_score: 0.56
  language: "en"
  document_id: "d9686f9481a362a8"

metadata:
  creation_date: "2025-08-17T13:08:21"
---
```html
<!-- page:1 -->
<!-- role:header -->
A!
Aalto University
OPEN ACCESS
<!-- role:paragraph -->
This is an electronic reprint of the original article.
This reprint may differ from the original in pagination and typographic detail.

Fam, Amin Moghimy; Lehtonen, Matti; Pourakbari-Kasmaei, Mahdi; Fotuhi-Firuzabad, Mahmud
Optimal Sizing of a Wind-PV Grid-Connected Hybrid System for Base Load- Helsinki Case

Published in:
2023 19th International Conference on the European Energy Market, EEM 2023

DOI:
10.1109/EEM58374.2023.10161955

Published: 01/01/2023

Document Version
Peer-reviewed accepted author manuscript, also known as Final accepted manuscript or Post-print

Please cite the original version:
Fam, A. M., Lehtonen, M., Pourakbari-Kasmaei, M., & Fotuhi-Firuzabad, M. (2023). Optimal Sizing of a Wind-PV Grid-Connected Hybrid System for Base Load- Helsinki Case. In _2023 19th International Conference on the European Energy Market, EEM 2023_ (International Conference on the European Energy Market, EEM; Vol. 2023 June). IEEE. [https://doi.org/10.1109/EEM58374.2023.10161955](https://doi.org/10.1109/EEM58374.2023.10161955)
<!-- role:footer -->

<!-- page:2 -->
<!-- role:heading level:1 -->
Optimal Sizing of a Wind-PV Grid-Connected
Hybrid System for Base Load– Helsinki Case
<!-- role:paragraph -->
Amin Moghimy Fam, Matti Lehtonen, Mahdi
Pourakbari-Kasmaei
Dept. Electrical Engineering and Automation
Aalto University
Espoo, Finland
amin.moghimyfam@aalto.fi, matti.lehtonen@aalto.fi,
mahdi.pourakbari@aalto.fi
<!-- role:paragraph -->
Abstract— In recent years, due to the goal of decarbonizing energy
systems, Renewable Energy Sources (RESs) have attracted
attention as the primary potential energy recourse in many
countries. However, the multi-scale behavior of these resources
has become of utmost importance. The large-scale connections
and the intermittent as well as variable characteristics of these
RESs cause challenges in maintaining a balance of power
generation and consumption. Furthermore, supplying base load
using RES is another challenge for system operators. Hybrid
RESs (HRESs), mixing wind, solar and grid, together with energy
storage might be a remedy via which the resources can
complement each other in some extent. In this paper, using
geographical data acquired from National Solar Radiation
Database and Matlab/Simulink, the output of each individual
solar panel and wind turbine in the Helsinki region are
calculated and used to optimally size an HRES to supply the base
load. The results indicate that an HRES has at least three times
less cost compared to a single RES system. Furthermore, it
can be seen that even in Finland, where there is not sufficient solar
radiation in winter, the sizing of the required energy storage system
reduces by at least 13.4 times when an HRES is used .
<!-- role:paragraph -->
*Index Terms—* Hybrid energy system, PV, Renewable energy
sources, Wind turbine.
<!-- role:heading level:1 -->
I. INTRODUCTION
<!-- role:paragraph -->
Renewable energy sources (RES), such as solar and wind,
offer a clean and economically competitive alternative to
conventional energy generation in which the cost of the energy is
produced using fossil fuels with large amounts of CO2 emission.
Producing energy with almost no CO2 emission using RES
aligns with the goal of becoming net-zero greenhouse gas
emission [1]. Although RESs possess a solution toward carbon
neutrality, intermittent power output and other associated
uncertainties raise many challenges for system planners and
operators. In this regard, Hybrid renewable energy systems
(HRESs) are considered to be useful as they introduce a
potential solution to overcome the challenges.
<!-- role:paragraph -->
Solar and wind power are among the most popular RESs.
These two sources can act in a complementary manner to
Mahmud Fotuhi-Firuzabad
Dept. Electrical Engineering
Sharif University of Technology
Tehran, Iran
fotuhi@sharif.edu
<!-- role:paragraph -->
smooth their intermittent power output. Hence, using a Wind-
PV HRES introduces a beneficial approach to enhancing the
economic and environmental sustainability of RESs and is
usually more efficient and reliable than a system with a
single renewable energy source [2] [3].
<!-- role:paragraph -->
In [4], to overcome the intermittent power output for
producing hydrogen, the application of Wind-PV HRESs,
including a battery energy storage system (BESS) was
introduced. The results show encouraging HRES efficiency
compared to similar experimental systems. In [5], an HRES
performance assessment procedure was proposed based on the
IEC-61400. The authors in [6] presented a detailed standalone
Wind-PV HRES sizing method and introduced a flexible
software based on the Levelized Power Supply Cost (LPSC)
algorithm and techno-economic analysis using object-oriented
programming. In [7], a review of hybrid energy systems
(HESs), consisting of both RES and conventional fossil fuel-
based generators, was presented. Also, a case study was
conducted to determine the most economical, and emission-
optimal configuration of a PV, wind, BESS and diesel
generator HES in a remote area. In [8], a sizing method for
Wind-PV HRES was proposed aiming to maximize the
annual ratio of the demand supplied by HRES to total demand
with a levelized cost of energy (LCOE) being equal to the grid
tariff, to meet both economic and environmental goals.
<!-- role:paragraph -->
The authors in [9] proposed an optimal techno-commercial
integration of PV, Wind, Biomass, and Variable Redox Flow
Battery (VRFB) in a microgrid to ensure the daily energy
demand. The simulations were performed under Hybrid
Optimization of Multiple Energy Resources (HOMER) and the
LCOE was used to validate the results. In [10], a study on a
6MW residential Wind-PV HRES was conducted where the
results show that the system is economical and provides an
opportunity for households to profit by implementing the
HRES. In [11], the optimal sizing of a grid-connected Wind-
PV HRES was investigated, aiming at minimizing the
difference between HRES daily power output and load and
maximizing HRES daily power generation. In [12], a genetic
algorithm-based heuristic approach was developed for the
<!-- role:heading level:2 -->
II. MATHEMATICAL MODEL OF HRES COMPONENTS
<!-- role:paragraph -->
In this section, the mathematical model of the grid connected
HRES is demonstrated. Firstly, the HRES, as the proposed
model, consists of PV panels, WTs, and BESS, as explained in
detail in the following sub-sections.
<!-- role:heading level:3 -->
A. PV Panel
<!-- role:paragraph -->
The behavior of a PV panel can be modeled as a nonlinear
current source with intrinsic series resistance. A PV module
consists of several solar cells, which are mainly p-n diode.
The output current of a PV cell is mostly dependent on
solar radiation (G^t_c) and cell temperature (T_c). Hence, the illustrated
model should provide an output of a PV cell considering both
these factors. Considering the PV panel equipped with a
maximum power point tracker (MPPT), the output power of a
PV panel can be calculated using [14]-[16]
<!-- role:equation -->
```
I_sc = I_{sc,ref} \frac{G_c}{G_{c,ref}}[1 + \alpha_I(T_c - T_{c,ref})]
```
<!-- role:equation -->
```
P_{pv,out} = V_{pv,mpp}I_{mpp}
```
<!-- role:paragraph -->
where,
<!-- role:equation -->
```
C_1 = I_a (1 - \frac{exp(-V_{oc}/V_a)}{V_{oc}/V_a})
```
<!-- role:equation -->
```
V_{oc} = V_{oc,ref} + \alpha_{voc}(T_c-T_{c,ref})
```
<!-- role:equation -->
```
V_{mpp} = V_{mpp,ref} + \alpha_{vmpp} \Delta T
```
<!-- role:equation -->
```
T_c = T_a + \frac{NOCT - 20}{800} G_c
```
<!-- role:paragraph -->
where I_mpp, V_mpp, and P_mpp are respectively the current,
voltage, and output power of the PV panel at the maximum
power point, I_sc, I_mpp, and V_mpp are respectively the
short circuit current, open circuit voltage, maximum point
current, and maximum point voltage at the reference point:
I_sc is the short circuit current temperature coefficient; α_I is the
open circuit voltage temperature coefficient; G_c is the solar
radiation; T_c and T_{c,ref} are the PV cell temperature and the PV cell
temperature at the reference point, respectively. T_a is the
ambient temperature, and NOCT is the nominal operating cell
temperature when PV panel operates under 800 W/m^2 of solar
radiation and at 20°C of ambient temperature. The utilized
parameters in the model and reference points are usually
provided in each PV panel datasheet. It should be noted that, in
this study, converters behavior has not been considered, and
only the maximum possible output of a PV panel is calculated.
<!-- role:paragraph -->
Using the proposed mathematical model for a PV panel, the
output power of PV panels can be estimated at given solar radiation
and ambient temperature at a given time (t) [9]
<!-- role:equation -->
```
P_{pv}(t) = P_{mpp} - I_{mpp}(t)V_{mpp}(t)
```
<!-- role:heading level:3 -->
B. Wind Turbine
<!-- role:paragraph -->
WT is another component of HRES. Characteristic curves
for WTs are given as power output versus the wind speed at the
hub height. The output power of a WT will be calculated using
the swept blade area (A), the air density (ρ), the wind velocity
(v), and the coefficient of power (Cp). The coefficient of power
can be formulated depending on the design factors (c1 - c6) as
<!-- role:equation -->
```
C_p(\lambda,\beta) = c_1 (\frac{c_2}{\lambda_i} - c_3\beta - c_4)e^{\frac{-c_5}{\lambda_i}} + c_6 \lambda
```
<!-- role:paragraph -->
where:
<!-- role:equation -->
```
\lambda = \frac{2 \pi R}{\nu}
```
<!-- role:equation -->
```
\frac{1}{\lambda_i} = \frac{1}{\lambda+0.035}-\frac{1}{1 + \beta^3.9}
```
<!-- role:paragraph -->
In this work, the mathematical model of the grid-
connected HRES is demonstrated. Firstly, the HRES, as the
proposed model, consists of PV panels, WTs, and BESS, as
explained in detail in the following sub-sections.
<!-- role:paragraph -->
The power of a WT can be expressed as (13).
<!-- role:equation -->
```
P_F(t) = 0.5 \rho A C_p(\lambda_i, \beta) \nu^3
```
<!-- role:paragraph -->
However, this output power is achieved only for a certain
range of wind speed. Generally, the WT output power is
presented by the data can be separated into areas as follows:
<!-- role:list type:ordered -->
-   1) The area where the wind speed is less than the cut-in
    wind speed (v_c,in). In this case, the WT generation is
    zero.
-   2) The area where wind speed is higher than the WT’s cut-
    in speed (v_c,in) and less than its nominal wind speed
    (v_n). In this case, WT generation is calculated using (13).
-   3) The area where wind speed is more than the WT’s
    nominal wind speed (v_n) and less than its cut-out wind
    speed (v_c, out). In this case, WT generates its nominal
    output power (P^*_n).
-   4) The area where wind speed is higher than the WT’s cut-
    out wind speed (v_c,out). In this case, WT generation does
    not generate any power.
<!-- role:paragraph -->
These conditions can be summarized as (14):
<!-- role:equation -->
```
P^F(t) = 0
```
<!-- role:equation -->
```
P^F(t) = \frac{1}{2}\rho \ A \ C_p(λ, β)v^3
```
<!-- role:equation -->
```
P^F_n
```
<!-- role:equation -->
```
P^F(t) = 0
```
<!-- role:heading level:3 -->
C. Battery Energy Storage System
<!-- role:paragraph -->
Due to intermittent and not fully controllable generation of
power sources in HRESs, an electrical energy storage system
(EESS) can be imperative to maintain the system operation.
Hence, an EESS is needed for the proposed HRES to be able to
inject adequate power from the grid to meet the base load.
Therefore, in this paper, it is referred to as a BESS.
<!-- role:paragraph -->
The main characteristic of a BESS is its state of charge
(SOC), which reflects the level of stored energy in a BESS
relative to its capacity and depth. Also, at a given time, the
total generation of the HRESs can be calculated as (15)
<!-- role:equation -->
```
P_{total}(t)  = P_{pv}(t) + P_{WT}(t)
```
<!-- role:paragraph -->
The SOC at each time SOC (t) for a 1-hour time step
depends on SOC (t-1) at a time interval before SOC
charging, power (P_{dich}(t)), and discharging power
(P_dich(t)). Hence, the SOC can be expressed as:
<!-- role:equation -->
```
SOC(t)= SOC(t-1)(1-σ)(1-\frac{η^c \ P_{cha}(t)}{E_{bat} ^ max}) + \frac{1}{η^d d} \frac{P_{dich}(t)}{E_{bat}^max}
```