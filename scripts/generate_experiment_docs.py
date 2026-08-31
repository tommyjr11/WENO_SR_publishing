#!/usr/bin/env python3
"""Generate reviewer-facing experiment documentation and launch wrappers."""

from __future__ import annotations

import json
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]

COMMON_METHODS = (
    "WENO5-JS--RK3, WENO5-Z--RK3, WENO5-SR--RK3, "
    "WENO5-SR-FP32--RK3, WENO7-JS--RK4, WENO7-Z--RK4, and WENO7-SR--RK4"
)

MODEL_NOTE = (
    "The learned methods use the selected WENO5-SR FP64 checkpoint at step 12,250, "
    "the independently trained WENO5-SR FP32 checkpoint at step 16,500, and the "
    "WENO7-SR FP64 checkpoint at step 16,750. Reflection-symmetrised inference is "
    "used in every learned reconstruction."
)

EXPERIMENTS = {
    "01_gste_long_time_advection": {
        "title": "Long-time advection of the GSTE profile",
        "equations": "One-dimensional scalar advection, u_t + u_x = 0.",
        "configuration": (
            "The periodic domain is [-1, 1], discretised with N=200 finite-volume cells. "
            "The solution is advanced to t=10 at CFL 0.6, corresponding to five complete "
            "domain traversals and 1,667 time steps. Initial and exact cell averages use "
            "15-point Gauss--Legendre quadrature."
        ),
        "initialization": (
            "The standard GSTE composite contains a narrow Gaussian packet on (-0.8,-0.6), "
            "a unit square wave on (-0.4,-0.2), a triangular pulse on (0,0.2), and a "
            "semi-ellipse on (0.4,0.6). The Gaussian and semi-ellipse use the three-point "
            "1:4:1 smoothing prescribed by the benchmark. The exact solution is the periodic "
            "translation u(x,t)=u_0(x-t)."
        ),
        "methods": COMMON_METHODS,
        "results": (
            "All learned methods remain stable for the full trajectory although GSTE is not "
            "a training profile. WENO7-SR gives the lowest L1 error, 2.342e-2, improving on "
            "WENO7-JS by 21.3% and WENO7-Z by 36.3%. The WENO5-SR FP32 model is the most "
            "accurate fifth-order variant with L1=3.645e-2."
        ),
        "plotting": (
            "The exact translated profile and all seven finite-volume solutions are drawn on "
            "the same axes. Method colours and line styles are fixed globally; no method-specific "
            "axis limits or filtering is applied."
        ),
        "config": {
            "domain": "[-1, 1]",
            "grid": "N=200",
            "boundary": "periodic",
            "cfl": 0.6,
            "t_end": 10.0,
            "initial_quadrature": "15-point Gauss-Legendre",
            "reference": "exact periodic finite-volume translation",
        },
        "run": [
            "python3 -u -m for_paper_results.run_gste --nx 200 --t-end 10 --init-quadrature 15 --weno5-cfl 0.6 --weno7-cfl 0.6 --methods weno5_js,weno5_sr_f64,weno5_sr_f32,weno7_js,weno7_sr_f64 --device \"$DEVICE\"",
            "python3 -u -m weno_z_borges_p2_results.run_gste --nx 200 --t-end 10 --cfl 0.6 --quadrature 15 --device \"$DEVICE\"",
        ],
        "plot": ["python3 -u -m weno_z_borges_p2_results.plot_gste"],
    },
    "02_sod_shock_tube": {
        "title": "Sod shock tube",
        "equations": "One-dimensional compressible Euler equations with gamma=1.4.",
        "configuration": (
            "The domain [0,1] uses N=100 finite-volume cells, transmissive boundaries, "
            "characteristic reconstruction, HLLC fluxes, CFL 0.8, and t=0.2. The auxiliary "
            "transverse extent contains ten identical cells but is not part of the stated "
            "one-dimensional resolution."
        ),
        "initialization": (
            "The discontinuity is at x=0.5. Primitive states are (rho,u,p)=(1,0,1) on the "
            "left and (0.125,0,0.1) on the right. Initial and exact finite-volume averages "
            "are evaluated with 15-point Gauss--Legendre quadrature."
        ),
        "methods": COMMON_METHODS,
        "results": (
            "WENO5-SR has the smallest density L1 error, 3.806e-3, which is 23.4% below "
            "WENO5-JS and 16.8% below WENO5-Z. WENO7-SR reaches 3.997e-3 and improves on both "
            "seventh-order classical baselines by approximately 19%."
        ),
        "plotting": (
            "Symbols denote numerical cell averages and a dense black curve denotes the exact "
            "pointwise Riemann solution. Enlarged panels show the post-rarefaction plateau, "
            "contact discontinuity, and shock with shared limits across methods."
        ),
        "config": {
            "domain": "[0, 1]",
            "grid": "N=100",
            "boundary": "transmissive",
            "cfl": 0.8,
            "t_end": 0.2,
            "riemann_solver": "HLLC",
            "reconstruction_space": "local characteristic",
            "initial_quadrature": "15-point Gauss-Legendre",
            "reference": "exact finite-volume Riemann solution",
        },
        "run": [
            "python3 -u -m for_paper_results.run_sod --nx 100 --ny 10 --cfl 0.8 --t-end 0.2 --init-quadrature 15 --out-tag N100_t020_cfl08 --device \"$DEVICE\"",
            "python3 -u -m weno_z_borges_p2_results.run_riemann_1d --problem sod --nx 100 --ny 10 --cfl 0.8 --out-tag N100_t020_cfl08 --device \"$DEVICE\"",
        ],
        "plot": [
            "python3 -u -m weno_z_borges_p2_results.plot_riemann_1d_zoom --problem sod --nx 100 --input-tag N100_t020_cfl08 --output-tag N100_t020_cfl08 --paper-style",
        ],
    },
    "03_lax_shock_tube": {
        "title": "Lax shock tube",
        "equations": "One-dimensional compressible Euler equations with gamma=1.4.",
        "configuration": (
            "The domain [-5,5] uses N=200 finite-volume cells, transmissive boundaries, "
            "characteristic reconstruction, HLLC fluxes, CFL 0.8, and final time t=1.3."
        ),
        "initialization": (
            "The initial interface is at x=0. Primitive states are "
            "(rho,u,p)=(0.445,0.698,3.528) on the left and (0.5,0,0.571) on the right. "
            "The reference is the exact finite-volume Riemann solution."
        ),
        "methods": COMMON_METHODS,
        "results": (
            "WENO7-SR gives the lowest density L1 error, 4.415e-3. WENO5-SR FP64 and FP32 "
            "are effectively coincident at 4.855e-3 and 4.856e-3, both clearly below the "
            "WENO5-JS and WENO5-Z errors."
        ),
        "plotting": (
            "Density, velocity, and pressure are compared to the exact solution. Insets use "
            "common windows around the contact and post-shock plateau; numerical values are "
            "shown as finite-volume samples rather than interpolated curves."
        ),
        "config": {
            "domain": "[-5, 5]",
            "grid": "N=200",
            "boundary": "transmissive",
            "cfl": 0.8,
            "t_end": 1.3,
            "riemann_solver": "HLLC",
            "reconstruction_space": "local characteristic",
            "reference": "exact finite-volume Riemann solution",
        },
        "run": [
            "for method in weno5_js weno5_sr_f64 weno5_sr_f32 weno7_js weno7_sr_f64; do python3 -u -m for_paper_results.run_shock_tube_benchmark --benchmark lax --method \"$method\" --nx 200 --ny 10 --cfl 0.8 --device \"$DEVICE\"; done",
            "python3 -u -m weno_z_borges_p2_results.run_riemann_1d --problem lax --nx 200 --ny 10 --cfl 0.8 --device \"$DEVICE\"",
        ],
        "plot": ["python3 -u -m weno_z_borges_p2_results.plot_lax_paper"],
    },
    "04_titarev_toro": {
        "title": "Titarev--Toro shock--entropy-wave interaction",
        "equations": "One-dimensional compressible Euler equations with gamma=1.4.",
        "configuration": (
            "The domain [-5,5] is evaluated at N=1001 and N=2000, with transmissive "
            "boundaries, characteristic HLLC fluxes, CFL 0.8, and t=5."
        ),
        "initialization": (
            "For x<-4.5 the primitive state is (1.515695,0.523346,1.805). Elsewhere it is "
            "(1+0.1 sin(20 pi x),0,1). The reference is a WENO5-JS calculation on N=10001, "
            "conservatively restricted to each reported mesh."
        ),
        "methods": COMMON_METHODS,
        "results": (
            "At N=1001, WENO5-SR has the lowest density L1 error, 1.682e-2. At N=2000, "
            "WENO7-SR is best at 1.849e-3, followed closely by WENO5-SR FP32 at 1.907e-3. "
            "The learned closures preserve substantially more of the post-shock entropy wave."
        ),
        "plotting": (
            "The full density profile and a post-shock enlargement use identical method "
            "colours at both resolutions. The high-resolution reference is restricted before "
            "error evaluation."
        ),
        "config": {
            "domain": "[-5, 5]",
            "grids": "N=1001 and N=2000",
            "boundary": "transmissive",
            "cfl": 0.8,
            "t_end": 5.0,
            "riemann_solver": "HLLC",
            "reference": "WENO5-JS, N=10001, conservative restriction",
        },
        "run": [
            "bash for_paper_results/run_titarev_toro_cfl08_ny10.sh",
            "python3 -u -m weno_z_borges_p2_results.run_titarev_toro --nx 1001 --ny 10 --cfl 0.8 --t-end 5 --device \"$DEVICE\"",
            "python3 -u -m weno_z_borges_p2_results.run_titarev_toro --nx 2000 --ny 10 --cfl 0.8 --t-end 5 --device \"$DEVICE\"",
        ],
        "plot": [
            "python3 -u -m weno_z_borges_p2_results.plot_titarev_toro --nx 1001 --ny 10",
            "python3 -u -m weno_z_borges_p2_results.plot_titarev_toro --nx 2000 --ny 10",
        ],
    },
    "05_isentropic_vortex": {
        "title": "Periodic isentropic-vortex convergence",
        "equations": "Two-dimensional compressible Euler equations with gamma=1.4.",
        "configuration": (
            "The periodic domain is [-10,10]^2. Meshes N=25, 50, 100, and 200 are evolved "
            "to t=2 at CFL 0.4. Initial and exact conservative cell averages use tensor-product "
            "15x15 Gauss--Legendre quadrature."
        ),
        "initialization": (
            "The free stream is rho=p=1 and (u,v)=(1,1). An isentropic vortex of strength "
            "beta=5 is centred at the origin and translates to (t,t). Periodic shortest-distance "
            "coordinates with period 20 define both the initial condition and exact solution."
        ),
        "methods": COMMON_METHODS,
        "results": (
            "The learned schemes retain high-order convergence. At N=200, WENO5-SR FP64 has "
            "rho-L1=2.701e-7 and WENO7-SR has 6.826e-8, both below their JS counterparts. "
            "WENO7-Z is the most accurate fine-grid smooth baseline, as expected from its "
            "critical-point construction."
        ),
        "plotting": (
            "Log--log L1 and L2 density errors are plotted against grid spacing with common "
            "reference slopes. Orders are computed between successive factor-two refinements."
        ),
        "config": {
            "domain": "[-10, 10]^2",
            "grids": "25^2, 50^2, 100^2, 200^2",
            "boundary": "periodic",
            "cfl": 0.4,
            "t_end": 2.0,
            "initial_quadrature": "15x15 Gauss-Legendre",
            "reference": "exact translated finite-volume vortex",
        },
        "run": [
            "python3 -u -m for_paper_results.run_vortex --grids 25,50,100,200 --cfl 0.4 --t-end 2 --quadrature 15 --out-tag vortex_cfl04 --device \"$DEVICE\"",
            "python3 -u -m weno_z_borges_p2_results.run_vortex --grids 25,50,100,200 --cfl 0.4 --t-end 2 --quadrature 15 --device \"$DEVICE\"",
        ],
        "plot": ["python3 -u -m weno_z_borges_p2_results.plot_vortex"],
    },
}

RIEMANN_CASES = {
    "06_riemann_c3": {
        "label": "C.3",
        "case": "c3",
        "t_end": 0.50,
        "states": "V1=(1.5,0,0,1.5); V2=(0.5323,1.206,0,0.3); V3=(0.138,1.206,1.206,0.029); V4=(0.5323,0,1.206,0.3)",
        "contours": "32 equally spaced density levels on [0.16,1.71]",
        "result": "WENO-SR preserves the large-scale wave geometry while exposing finer contact and shear-layer structure than the order-matched JS solutions.",
    },
    "07_riemann_c4": {
        "label": "C.4",
        "case": "c4",
        "t_end": 0.25,
        "states": "V1=(1.1,0,0,1.1); V2=(0.5065,0.8939,0,0.35); V3=(1.1,0.8939,0.8939,1.1); V4=(0.5065,0,0.8939,0.35)",
        "contours": "19 equally spaced density levels on [0.64,1.92]",
        "result": "All methods reproduce the symmetric lens-shaped interaction; the learned closures retain sharper internal contacts without changing the principal wave topology.",
    },
    "08_riemann_c5": {
        "label": "C.5",
        "case": "c5",
        "t_end": 0.23,
        "states": "V1=(1,-0.75,-0.5,1); V2=(2,-0.75,0.5,1); V3=(1,0.75,0.5,1); V4=(3,0.75,-0.5,1)",
        "contours": "21 equally spaced density levels on [1.10,3.90]",
        "result": "The learned solutions sharpen the interacting slip lines while remaining positive and free of non-finite states at the prescribed final time.",
    },
    "09_riemann_c6": {
        "label": "C.6",
        "case": "c6",
        "t_end": 0.30,
        "states": "V1=(1,0.75,-0.5,1); V2=(2,0.75,0.5,1); V3=(1,-0.75,0.5,1); V4=(3,-0.75,-0.5,1)",
        "contours": "25 equally spaced density levels on [0.54,2.94]",
        "result": "The WENO-SR fields retain the common large-scale solution and resolve additional shear-layer detail without instability.",
    },
}

for key, case in RIEMANN_CASES.items():
    EXPERIMENTS[key] = {
        "title": f"Two-dimensional Riemann configuration {case['label']}",
        "equations": "Two-dimensional compressible Euler equations with gamma=1.4.",
        "configuration": (
            f"The domain [0,1]^2 uses a 400x400 finite-volume mesh, outflow boundaries, "
            f"characteristic HLLC fluxes, CFL 0.4, and t={case['t_end']:.2f}. Initial "
            "discontinuities are aligned with x=0.5 and y=0.5."
        ),
        "initialization": (
            "States V1--V4 occupy the upper-right, upper-left, lower-left, and lower-right "
            f"quadrants: {case['states']}. Tensor-product 15x15 Gauss--Legendre quadrature "
            "constructs the initial finite-volume averages."
        ),
        "methods": (
            "WENO5-JS--RK3, WENO5-SR--RK3, WENO5-SR-FP32--RK3, "
            "WENO7-JS--RK4, WENO7-Z--RK4, and WENO7-SR--RK4"
        ),
        "results": case["result"],
        "plotting": (
            "Pressure is shown in colour and density as black contours. Every method in this "
            f"configuration shares one pressure normalisation and {case['contours']}. "
            + ("Plot coordinates are translated by (0.5,0.5)." if case["case"] == "c4" else "The physical coordinates are retained.")
        ),
        "config": {
            "case": case["label"],
            "domain": "[0, 1]^2",
            "grid": "400x400",
            "boundary": "outflow",
            "cfl": 0.4,
            "t_end": case["t_end"],
            "riemann_solver": "HLLC",
            "reconstruction_space": "local characteristic",
            "initial_quadrature": "15x15 Gauss-Legendre",
            "density_contours": case["contours"],
        },
        "run": [
            f"python3 -u -m for_paper_results.run_quadrant --case {case['case']} --nx 400 --ny 400 --cfl 0.4 --t-end {case['t_end']} --init-quadrature 15 --device \"$DEVICE\"",
            f"python3 -u -m weno_z_borges_p2_results.run_quadrant --case {case['case']} --nx 400 --ny 400 --cfl 0.4 --t-end {case['t_end']} --init-quadrature 15 --device \"$DEVICE\"",
        ],
        "plot": ["python3 -u -m weno_z_borges_p2_results.plot_riemann --hybrid-fields-only"],
    }

EXPERIMENTS.update(
    {
        "10_double_mach_reflection": {
            "title": "Double Mach reflection",
            "equations": "Two-dimensional compressible Euler equations with gamma=1.4.",
            "configuration": (
                "The standard double-Mach reflection problem is computed on a 1200x300 mesh "
                "to t=0.2 with CFL 0.4, characteristic reconstruction, HLLC fluxes, 15-point "
                "finite-volume initialisation, and no ENO cutoff or state repair."
            ),
            "initialization": (
                "A Mach-10 oblique shock initially intersects the lower wall at x=1/6 and "
                "forms a 60-degree angle with that wall. The moving post-shock state is imposed "
                "on the upper boundary consistently with the exact shock motion; the lower wall "
                "uses the standard inflow/reflection split."
            ),
            "methods": COMMON_METHODS,
            "results": (
                "All seven methods reach t=0.2 without non-finite values or negative density or "
                "pressure. WENO7-SR resolves the longest hierarchy of coherent secondary roll-ups "
                "along the slip line and near the wall while preserving the baseline shock geometry."
            ),
            "plotting": (
                "Separate WENO5 and WENO7 figures pair the full density field with the same "
                "2.05<=x<=2.85, 0<=y<=0.55 enlargement. All methods share one density colour "
                "scale and 29 uniformly spaced black density contours. The third-party comparison "
                "image used for discussion in the draft is intentionally not redistributed here."
            ),
            "config": {
                "domain": "[0, 4] x [0, 1]",
                "grid": "1200x300",
                "boundary": "standard double-Mach inflow/reflection",
                "cfl": 0.4,
                "t_end": 0.2,
                "riemann_solver": "HLLC",
                "reconstruction_space": "local characteristic",
                "initial_quadrature": "15-point Gauss-Legendre",
                "density_contours": "29 common equally spaced levels",
            },
            "run": [
                "python3 -u run_double_mach_compare.py --model teacherfree_lab_weno5_v20_distance_balanced/runs/apost_weno5_v20_distance_balanced_cfl05_200k/checkpoints/model_step_012250.npz --nx 1200 --ny 300 --cfl 0.4 --t-end 0.2 --init-quadrature 15 --weno-space characteristic --riemann-solver hllc --no-eno-cutoff --run-weno5 --no-run-weno7 --out-dir plots/WENO5_MLP/weno_double_reflective_1200 --device $DEVICE",
                "python3 -u -m for_paper_results.run_double_mach --nx 1200 --ny 300 --cfl 0.4 --t-end 0.2 --init-quadrature 15 --methods weno5_sr_f64,weno5_sr_f32,weno7_js,weno7_sr_f64 --device \"$DEVICE\"",
                "python3 -u -m weno_z_borges_p2_results.run_double_mach --methods weno5_z_p2,weno7_z_p3 --nx 1200 --ny 300 --cfl 0.4 --t-end 0.2 --init-quadrature 15 --device \"$DEVICE\"",
            ],
            "plot": ["python3 -u -m weno_z_borges_p2_results.plot_double_mach"],
        },
        "11_shock_bubble_2d_ma122": {
            "title": "Planar Mach-1.22 shock--helium-bubble interaction",
            "equations": "Two-dimensional single-gamma compressible Euler equations with gamma=1.4.",
            "configuration": (
                "The domain [0,0.225]x[0,0.089] uses 1000x396 cells, transmissive boundaries, "
                "characteristic HLLC fluxes, CFL 0.4, and t=6.0e-4. The comparison reference is "
                "an independent WENO7-JS--RK4 solution on 2000x791 cells."
            ),
            "initialization": (
                "A planar Mach-1.22 shock starts at x=0.005. Air has rho=1.29 and p=101325; "
                "the circular helium region has rho=0.214, the same pressure, radius 0.025, and "
                "centre (0.035,0.0445). Rankine--Hugoniot relations determine the post-shock "
                "density and pressure, with u_post=110.6273. Cell averages use 15x15 quadrature."
            ),
            "methods": COMMON_METHODS,
            "results": (
                "The WENO7-SR vortex-core locations and rolled-up interface agree closely with "
                "the 2000x791 reference. It has the smallest local density L1 error on all three "
                "reported cuts, reducing WENO7-JS errors by 65.2%, 51.1%, and 35.5%."
            ),
            "plotting": (
                "The full-domain mock-schlieren panels use one shared normalisation. Density cuts "
                "are reported at y=0.01900, y=0.04525, and x=0.12566; the fine reference is "
                "conservatively restricted before L1 evaluation."
            ),
            "config": {
                "domain": "[0, 0.225] x [0, 0.089]",
                "grid": "1000x396",
                "reference_grid": "2000x791 WENO7-JS--RK4",
                "shock_mach": 1.22,
                "boundary": "transmissive",
                "cfl": 0.4,
                "t_end": 0.0006,
                "riemann_solver": "HLLC",
                "initial_quadrature": "15x15 Gauss-Legendre",
            },
            "run": [
                "MAIN_DIR=shockbubble_t0006_cfl04_server/results/raw/shockbubble_t0006_cfl04/N1000x396",
                "REFERENCE_DIR=shockbubble_t0006_cfl04_server/results/raw/shockbubble_t0006_cfl04/reference_weno7_N2000x791",
                "mkdir -p \"$MAIN_DIR\" \"$REFERENCE_DIR\"",
                "for method in weno5_js weno5_sr_f64 weno5_sr_f32 weno7_js weno7_sr_f64; do python3 -u -m for_paper_results.run_weno5_shockbubble --method \"$method\" --nx 1000 --ny 396 --cfl 0.4 --t-end 0.0006 --report-interval 200 --out-dir \"$MAIN_DIR\" --device \"$DEVICE\"; done",
                "python3 -u -m for_paper_results.run_weno5_shockbubble --method weno7_js --nx 2000 --ny 791 --cfl 0.4 --t-end 0.0006 --report-interval 200 --out-dir \"$REFERENCE_DIR\" --device \"$DEVICE\"",
                "python3 -u -m weno_z_borges_p2_results.run_shockbubble_2d --case ma122 --method weno5_z_p2 --nx 1000 --ny 396 --cfl 0.4 --device \"$DEVICE\"",
                "python3 -u -m weno_z_borges_p2_results.run_shockbubble_2d --case ma122 --method weno7_z_p3 --nx 1000 --ny 396 --cfl 0.4 --device \"$DEVICE\"",
            ],
            "plot": ["python3 -u -m weno_z_borges_p2_results.plot_shockbubble_2d --case ma122"],
        },
        "12_shock_bubble_2d_ma30": {
            "title": "Planar Mach-3 shock--helium-bubble interaction",
            "equations": "Two-dimensional single-gamma compressible Euler equations with gamma=1.4.",
            "configuration": (
                "The domain [0,0.225]x[0,0.089] uses 1000x396 cells, transmissive boundaries, "
                "characteristic HLLC fluxes, CFL 0.4, and t=1.0e-4. The comparison reference is "
                "an independent WENO7-JS--RK4 solution on 2000x791 cells."
            ),
            "initialization": (
                "A planar Mach-3 shock starts at x=0.005. Air has rho=1.29 and p=101325; the "
                "helium bubble has rho=0.214, radius 0.025, and centre (0.035,0.0445). The "
                "post-shock speed is 736.911 and Rankine--Hugoniot relations set rho and p. "
                "Cell averages use tensor-product 15x15 quadrature."
            ),
            "methods": COMMON_METHODS,
            "results": (
                "WENO7-SR is the most accurate 1000x396 method on all three selected cuts and "
                "reduces the WENO7-JS local density errors by 56.9%, 25.0%, and 31.9%. Both "
                "WENO5-SR precisions also improve on WENO5-Z on these cuts."
            ),
            "plotting": (
                "The full-domain mock-schlieren panels share one normalisation. Density cuts are "
                "taken at x=0.06750, x=0.10991, and y=0.02056 through complex interaction regions."
            ),
            "config": {
                "domain": "[0, 0.225] x [0, 0.089]",
                "grid": "1000x396",
                "reference_grid": "2000x791 WENO7-JS--RK4",
                "shock_mach": 3.0,
                "boundary": "transmissive",
                "cfl": 0.4,
                "t_end": 0.0001,
                "riemann_solver": "HLLC",
                "initial_quadrature": "15x15 Gauss-Legendre",
            },
            "run": [
                "MAIN_DIR=shockbubble_ma3_t0001_cfl04_server/results/raw/shockbubble_ma3_t0001_cfl04/N1000x396",
                "REFERENCE_DIR=shockbubble_ma3_t0001_cfl04_server/results/raw/shockbubble_ma3_t0001_cfl04/reference_weno7_N2000x791",
                "mkdir -p \"$MAIN_DIR\" \"$REFERENCE_DIR\"",
                "for method in weno5_js weno5_sr_f64 weno5_sr_f32 weno7_js weno7_sr_f64; do python3 -u shockbubble_ma3_t0001_cfl04_server/for_paper_results/run_weno5_shockbubble.py --method \"$method\" --nx 1000 --ny 396 --cfl 0.4 --t-end 0.0001 --report-interval 200 --out-dir \"$MAIN_DIR\" --device \"$DEVICE\"; done",
                "python3 -u shockbubble_ma3_t0001_cfl04_server/for_paper_results/run_weno5_shockbubble.py --method weno7_js --nx 2000 --ny 791 --cfl 0.4 --t-end 0.0001 --report-interval 200 --out-dir \"$REFERENCE_DIR\" --device \"$DEVICE\"",
                "python3 -u -m weno_z_borges_p2_results.run_shockbubble_2d --case ma30 --method weno5_z_p2 --nx 1000 --ny 396 --cfl 0.4 --device \"$DEVICE\"",
                "python3 -u -m weno_z_borges_p2_results.run_shockbubble_2d --case ma30 --method weno7_z_p3 --nx 1000 --ny 396 --cfl 0.4 --device \"$DEVICE\"",
            ],
            "plot": ["python3 -u -m weno_z_borges_p2_results.plot_shockbubble_2d --case ma30"],
        },
        "13_shock_bubble_3d_ma30": {
            "title": "Three-dimensional Mach-3 shock--bubble interaction",
            "equations": "Three-dimensional single-material compressible Euler equations with gamma=1.4.",
            "configuration": (
                "The domain [0,0.225]x[0,0.089]x[0,0.089] uses 224x88x88 cells, "
                "transmissive boundaries, characteristic reconstruction, EVILIN fluxes, CFL 0.25, "
                "and t=1.0e-4. Classical JS context is also available on 448x176x176. The reference "
                "is WENO7--ADER4 on 1120x440x440 with the same EVILIN flux."
            ),
            "initialization": (
                "The shock starts at x=0.005. The post-shock state is "
                "(rho,u,v,w,p)=(4.975714,736.911,0,0,1.047025e6), air is "
                "(1.29,0,0,0,101325), and the spherical light region is "
                "(0.214,0,0,0,101325), centred at (0.035,0.0445,0.0445) with radius 0.025. "
                "Initial conservative cell averages use 3x3x3 Gauss quadrature."
            ),
            "methods": COMMON_METHODS,
            "results": (
                "WENO7-SR has the smallest 224x88x88 integrated density error on all three "
                "central-plane cuts. At x=0.085 it reduces the WENO7-JS error by 21.9%; the FP64 "
                "and FP32 WENO5-SR fields are visually and quantitatively very close. The learned "
                "improvements are concentrated around the rolled-up, under-resolved interface."
            ),
            "plotting": (
                "Volume renderings and central-z mock-schlieren use common transfer functions and "
                "normalisations. Density profiles at x=0.085, 0.097, and 0.114 are compared to "
                "the high-resolution reference; 2:1 data are symmetrically averaged and the aligned "
                "5:1 reference uses coincident cell centres."
            ),
            "config": {
                "domain": "[0, 0.225] x [0, 0.089] x [0, 0.089]",
                "grid": "224x88x88",
                "context_grids": "448x176x176 WENO5/7-JS",
                "reference_grid": "1120x440x440 WENO7--ADER4",
                "shock_mach": 3.0,
                "boundary": "transmissive",
                "cfl": 0.25,
                "t_end": 0.0001,
                "riemann_solver": "EVILIN",
                "initial_quadrature": "3x3x3 Gauss-Legendre",
            },
            "run": [
                "python3 -u -m warp_weno5_3d_rk3.run_shockbubble_ma3 --nx 224 --ny 88 --nz 88 --cfl 0.25 --t-end 0.0001 --device \"$DEVICE\" --out-dir runs/weno5_js",
                "python3 -u -m warp_weno5_3d_rk3.run_shockbubble_ma3_mlp --nx 224 --ny 88 --nz 88 --cfl 0.25 --t-end 0.0001 --device \"$DEVICE\" --out-dir runs/weno5_sr_f64",
                "python3 -u -m warp_weno5_3d_rk3.run_shockbubble_ma3_mlp_f32 --nx 224 --ny 88 --nz 88 --cfl 0.25 --t-end 0.0001 --device \"$DEVICE\" --out-dir runs/weno5_sr_f32",
                "python3 -u -m warp_weno7_3d_rk4.run_shockbubble_ma3 --nx 224 --ny 88 --nz 88 --cfl 0.25 --t-end 0.0001 --device \"$DEVICE\" --out-dir runs/weno7_js",
                "python3 -u -m warp_weno7_3d_rk4.run_shockbubble_ma3 --nx 224 --ny 88 --nz 88 --cfl 0.25 --t-end 0.0001 --device \"$DEVICE\" --model teacherfree_lab_weno7_rk4_distance_balanced_fast/runs/apost_weno7_rk4_distance_balanced_fast_4090_200k/checkpoints/model_step_016750.npz --out-dir runs/weno7_sr_f64",
                "python3 -u -m weno_z_borges_p2_results.run_shockbubble_3d --method weno5_z_p2 --nx 224 --ny 88 --nz 88 --cfl 0.25 --t-end 0.0001 --device \"$DEVICE\" --out-dir runs/weno5_z",
                "python3 -u -m weno_z_borges_p2_results.run_shockbubble_3d --method weno7_z_p3 --nx 224 --ny 88 --nz 88 --cfl 0.25 --t-end 0.0001 --device \"$DEVICE\" --out-dir runs/weno7_z",
            ],
            "plot": [
                "python3 -u -m weno_z_borges_p2_results.plot_shockbubble_3d",
                "python3 -u -m weno_z_borges_p2_results.plot_shockbubble_3d_linecuts --x 0.085 0.097 0.114 --z 0.0445",
            ],
        },
        "14_mixed_precision_timing": {
            "title": "Mixed-precision WENO5 inference timing",
            "equations": "Two-dimensional Euler configuration C.4 used as a fixed timing workload.",
            "configuration": (
                "The 400x400 C.4 calculation is timed after Warp compilation, model loading, "
                "initialisation, diagnostics, plotting, and file output are excluded. Each method "
                "is warmed up and then measured three times."
            ),
            "initialization": (
                "The numerical problem is identical to experiment 07. Only the MLP parameter and "
                "activation precision changes: finite-volume states, characteristic transforms, "
                "HLLC fluxes, and SSPRK3 updates remain FP64 in all variants."
            ),
            "methods": "WENO5-JS, WENO5-SR with an FP64 MLP, and WENO5-SR with an FP32 MLP.",
            "results": (
                "Mean wall times are 6.605+/-0.032 s, 39.333+/-0.280 s, and 14.253+/-0.072 s "
                "for JS, FP64 MLP, and FP32 MLP. FP32 therefore accelerates learned inference by "
                "2.76x while preserving the FP64 finite-volume solver."
            ),
            "plotting": "No figure is used; the three-repeat statistics are reported as a LaTeX table.",
            "config": {
                "problem": "Riemann C.4",
                "grid": "400x400",
                "warmup": "one untimed run per method",
                "repeats": 3,
                "excluded_costs": "compilation, loading, initialization, diagnostics, plotting, output",
            },
            "run": ["python3 -u -m for_paper_results.run_weno5_timing --repeats 3 --methods weno5_js,weno5_sr_f64,weno5_sr_f32 --device \"$DEVICE\""],
            "plot": ["printf '%s\\n' 'This experiment is reported as tables/weno5_precision_timing.tex; no plot is generated.'"],
        },
    }
)


def yaml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def write_config(path: Path, key: str, item: dict[str, object]) -> None:
    lines = [f"experiment_id: {yaml_scalar(key)}", f"title: {yaml_scalar(item['title'])}"]
    for field, value in item["config"].items():
        lines.append(f"{field}: {yaml_scalar(value)}")
    lines.extend(
        [
            "selected_checkpoints:",
            '  weno5_sr_fp64: "../../models/weno5_sr_fp64_step012250.npz"',
            '  weno5_sr_fp32: "../../models/weno5_sr_fp32_step016500.npz"',
            '  weno7_sr_fp64: "../../models/weno7_sr_fp64_step016750.npz"',
            "reflection_symmetrised_inference: true",
            "eno_cutoff: false",
            "raw_data_in_release: false",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(path: Path, item: dict[str, object]) -> None:
    text = f"""# {item['title']}

## Purpose

{item['equations']} This benchmark is part of the out-of-distribution validation suite for the shared learned smoothness-ratio closure.

## Configuration

{item['configuration']}

## Initial condition

{item['initialization']}

## Methods

{item['methods']}. {MODEL_NOTE}

## Results

{item['results']}

The compact numerical values used by the manuscript are retained in [`tables/`](tables/), while the publication-ready result is retained in [`figures/`](figures/).

## Plot construction

{item['plotting']}

The plotting scripts operate on regenerated local outputs and do not alter the archived publication figure. Vector PDF is retained whenever available.

## Reproduction

From this experiment directory:

```bash
DEVICE=cuda bash run.sh
bash plot.sh
```

The full calculation may require a CUDA GPU and substantial wall time. The wrappers redirect every solver-defined output root to this experiment's ignored `runs/` directory; the immutable code snapshot is therefore not populated with regenerated fields.

## Provenance and data policy

[`provenance.json`](provenance.json) records the original path, SHA-256 digest, and size of every archived figure and table. Raw multidimensional fields are deliberately excluded from this GitHub snapshot. The solver and plotting sources needed to regenerate them are preserved verbatim under [`../../code/snapshot/`](../../code/snapshot/).
"""
    path.write_text(text, encoding="utf-8")


def write_shell(path: Path, commands: list[str]) -> None:
    body = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'REPOSITORY="$(cd "$HERE/../.." && pwd)"',
        'SNAPSHOT="$REPOSITORY/code/snapshot"',
        'DEVICE="${DEVICE:-cuda}"',
        'cd "$SNAPSHOT"',
        'export PYTHONPATH="$SNAPSHOT${PYTHONPATH:+:$PYTHONPATH}"',
        'RUN_ROOT="$HERE/runs"',
        'mkdir -p "$RUN_ROOT"',
        '',
        'redirect_output() {',
        '  local source="$1"',
        '  local target="$2"',
        '  mkdir -p "$(dirname "$source")" "$target"',
        '  if [[ -e "$source" && ! -L "$source" ]]; then',
        '    printf "refusing to replace existing generated directory: %s\\n" "$source" >&2',
        '    exit 2',
        '  fi',
        '  ln -sfn "$target" "$source"',
        '}',
        '',
        'redirect_output "$SNAPSHOT/for_paper_results/raw" "$RUN_ROOT/for_paper_results/raw"',
        'redirect_output "$SNAPSHOT/for_paper_results/figures" "$RUN_ROOT/for_paper_results/figures"',
        'redirect_output "$SNAPSHOT/for_paper_results/tables" "$RUN_ROOT/for_paper_results/tables"',
        'redirect_output "$SNAPSHOT/weno_z_borges_p2_results/raw" "$RUN_ROOT/weno_z_borges_p2_results/raw"',
        'redirect_output "$SNAPSHOT/weno_z_borges_p2_results/figures" "$RUN_ROOT/weno_z_borges_p2_results/figures"',
        'redirect_output "$SNAPSHOT/weno_z_borges_p2_results/tables" "$RUN_ROOT/weno_z_borges_p2_results/tables"',
        'redirect_output "$SNAPSHOT/shockbubble_t0006_cfl04_server/results" "$RUN_ROOT/shockbubble_t0006_cfl04_server/results"',
        'redirect_output "$SNAPSHOT/shockbubble_ma3_t0001_cfl04_server/results" "$RUN_ROOT/shockbubble_ma3_t0001_cfl04_server/results"',
        'redirect_output "$SNAPSHOT/plots" "$RUN_ROOT/plots"',
        'redirect_output "$SNAPSHOT/runs" "$RUN_ROOT/solver_runs"',
        "",
        *commands,
        "",
    ]
    path.write_text("\n".join(body), encoding="utf-8")
    path.chmod(0o755)


def main() -> None:
    expected = {f"{index:02d}_" for index in range(1, 15)}
    if len(EXPERIMENTS) != 14 or any(not key.startswith(tuple(expected)) for key in EXPERIMENTS):
        raise RuntimeError("experiment manifest must contain exactly the 14 release experiments")
    for key in sorted(EXPERIMENTS):
        item = EXPERIMENTS[key]
        root = REPOSITORY / "experiments" / key
        root.mkdir(parents=True, exist_ok=True)
        (root / "figures").mkdir(exist_ok=True)
        (root / "tables").mkdir(exist_ok=True)
        if key == "14_mixed_precision_timing":
            (root / "figures" / "README.md").write_text(
                "# Figure policy\n\n"
                "This benchmark is reported as a timing table in the manuscript; "
                "no derived figure is used or archived.\n",
                encoding="utf-8",
            )
        write_readme(root / "README.md", item)
        write_config(root / "config.yaml", key, item)
        write_shell(root / "run.sh", item["run"])
        write_shell(root / "plot.sh", item["plot"])
    print(f"generated documentation and wrappers for {len(EXPERIMENTS)} experiments")


if __name__ == "__main__":
    main()
