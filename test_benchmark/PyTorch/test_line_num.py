import os
for f in os.listdir("./"):
    ff = os.path.join("./", f)
    if os.path.isdir(ff) and os.path.exists(os.path.join(ff,
                                                         "pd_model_script")):
        print(
            ff,
            len(
                open(os.path.join(
                    ff, "pd_model_script/x2paddle_code.py")).readlines()) - 14)
