import os
import glob
import os

# ****************************************************************
# training
# New training
cmd0 = 'py deepMedicRun -model ./examples/configFiles/deepMedicBM/model/modelConfig_wide1_deeper.cfg  -train ./examples/configFiles/deepMedicBM/train/trainConfigwideAll.cfg  -dev cuda1'

#Recommend fine tuning
cmd1 = 'py deepMedicRun -model ./examples/configFiles/deepMedicBM/model/modelConfig_wide1_deeper.cfg  -train ./examples/configFiles/deepMedicBM/train/trainConfigwideAll.cfg  -load  ./examples/output/saved_models/singlePathTrainingModel/deepMedicWide1.singlePathTrainingModel.final.model.ckpt -dev cuda1'

#test
cmd2 = 'py deepMedicRun -model ./examples/configFiles/deepMedicBM/model/modelConfig_wide1_deeper.cfg  -test ./examples/configFiles/deepMedicBM/test/testConfig.cfg -load  ./examples/output/saved_models/singlePathTrainingModel/deepMedicWide1.singlePathTrainingModel.final.model.ckpt -dev cuda1'

cmds = [cmd2]

for cmd in cmds:
    os.system(cmd)