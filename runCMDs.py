import os
import glob
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# ****************************************************************
# training
#
cmd1 = 'py deepMedicRun -model ./examples/configFiles/deepMedicBM/model/modelConfig_wide1_deeper.cfg  -train ./examples/configFiles/deepMedicBM/train/trainConfigwideAll.cfg  -dev cuda1'
cmd2 = 'py deepMedicRun -model ./examples/configFiles/deepMedicBM/model/modelConfig_wide1_deeper.cfg  -test ./examples/configFiles/deepMedicBM/test/testConfig.cfg -load  ./examples/output/saved_models/HighSensitivityAllRmsPropSI/deepMedicWide1.HighSensitivityAllRmsPropSI.final.model.ckpt -dev cuda1'

cmds = [cmd2]

for cmd in cmds:
    os.system(cmd)