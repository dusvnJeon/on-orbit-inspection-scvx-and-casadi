from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(r"C:\Users\what_\OneDrive\바탕 화면\대학원\KROS 논문확장_홍준기연구원님")
ROCKIT_DIR = PROJECT_ROOT / "rockit"
if str(ROCKIT_DIR) not in sys.path:
    sys.path.insert(0, str(ROCKIT_DIR))

from dynamics_residuals import load_alipddp_translation_history, save_rockit_style_csv
from plot_solution_histories import plot_translational_velocity_control_histories, read_translational_solution


STATE_HISTORY_CSV = Path(
    r"C:\Users\what_\Downloads\ALIPDDP_back-main\visualizer\inspection\4phase"
    r"\4phase_translation_hs_sto_original_ver2_state_history.csv"
)

ROCKIT_FORMAT_CSV = Path(
    r"C:\Users\what_\OneDrive\바탕 화면\대학원\KROS 논문확장_홍준기연구원님"
    r"\compare_data\alipddp_original_ver2_300_rockit_format.csv"
)

FIGURE9_OUTPUT = Path(
    r"C:\Users\what_\OneDrive\바탕 화면\대학원\KROS 논문확장_홍준기연구원님"
    r"\compare_data\figure9_velocity_control_alipddp_ver2_300.png"
)


def main() -> None:
    phases = load_alipddp_translation_history(STATE_HISTORY_CSV)
    try:
        csv_path = save_rockit_style_csv(phases, ROCKIT_FORMAT_CSV)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot overwrite {ROCKIT_FORMAT_CSV}. Close the CSV in Excel, VSCode preview, "
            "or any process that may be locking it, then run this script again."
        ) from exc

    rows = read_translational_solution(csv_path)
    figure_path = plot_translational_velocity_control_histories(rows, FIGURE9_OUTPUT)

    print(f"Saved Rockit-format CSV: {csv_path}")
    print(f"Saved Figure 9 plot: {figure_path}")


if __name__ == "__main__":
    main()
