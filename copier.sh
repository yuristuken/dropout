REMOTE_PATH='/root/dropout'
PATH_PREFIX=$REMOTE_PATH'/results_20150511/784,1024,1024,10_activations=TanhActivationFunction,TanhActivationFunction,SigmoidActivationFunction_dropout=0.2,0.5,0.5_dropoutType=2_extraParam='
PATH_SUFFIX='_maxNorm=4.0_epochs=50_learningRate=0.001_momentum=0.95_regCoefficient=0.0_batchSize=50'

PARAM='0.5'
MACHINE='phoebe2-3'

mkdir -- $PARAM
cd -- "$PARAM"
#mkdir 'nodropout'
#cd 'nodropout'
scp root@$MACHINE.stuken.me:$PATH_PREFIX$PARAM$PATH_SUFFIX'/validation_corrects' .
scp root@$MACHINE.stuken.me:$PATH_PREFIX$PARAM$PATH_SUFFIX'/training_corrects' .
scp root@$MACHINE.stuken.me:$REMOTE_PATH'/log_type=2_'$PARAM .
cd ..
