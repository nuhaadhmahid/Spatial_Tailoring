# Plotting.py
import os
import pandas as pd

import Utils

if __name__ == "__main__":

    case_name = "r1_incr_size"
    directory = Utils.Directory(case_name)

    # plotting all cases together
    all_responses = {}
    for case_number in range(0, 4):
        response_file = os.path.join(
            directory.case_folder,
            "data",
            f"{case_number}_fairing_response.xlsx",
        )
        if os.path.exists(response_file):
            df = pd.read_excel(response_file)
            all_responses[case_number] = dict(
                list(zip(df.columns, [df[col].values for col in df.columns]))
            )

    Utils.Plots.fairing_response(
        all_responses,
        save_path=os.path.join(
            directory.case_folder,
            "fig",
            f"all_fairing_responses.png",
        ),
        show=True,
    )
