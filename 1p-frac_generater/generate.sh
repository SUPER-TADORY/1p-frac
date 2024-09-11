BASE_SAVEDIR="/groups/gag51404/user/tadokoro/OFDB/OFDB/code/data_rendering/_tmp" # directory to save 1p-frac data
SIGMA=4.0 # parameter to control shape complexity
DELTA=0.1 # parameter to control shape variance
SAMPLE=1000 # Number of samples from shape distribution controlled by delta

python render_from_oneparam_mp.py --img_basesavedir $BASE_SAVEDIR \
                                  --sigma $SIGMA \
                                  --delta $DELTA \
                                  --sample $SAMPLE