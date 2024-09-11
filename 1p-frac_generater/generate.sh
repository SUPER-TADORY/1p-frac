BASE_SAVEDIR="/path/to/savedir"
SIGMA=4.0
DELTA=0.1 
SAMPLE=1000

python render_from_oneparam_mp.py --img_basesavedir $BASE_SAVEDIR \
                                  --sigma $SIGMA \
                                  --delta $DELTA \
                                  --sample $SAMPLE