import os
save_model_name='test_'
#os.system("echo sed -i -r \\'s/\\.*\\(.{150}\\)/\1/g\\' "+save_model_name+"latest_run.log"+" && echo sed -i -r \\'s/[\\\\]//g\\' "+save_model_name+"latest_run.log")
os.system("echo sed -i -r \\'s/.*\\(.{150}\\)/\1/g\\' "+save_model_name+"latest_run.log")
os.system("echo sed -i -r \\'s/[\\]//g\\' "+save_model_name+"latest_run.log")
