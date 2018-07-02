import time, logging, sys, os, re
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("getmodels.py: " + __name__)

def getmodels(modeldir, modelnum):
    files = [f for f in os.listdir(modeldir) if os.path.isfile(os.path.join(modeldir, f))]
    # model.npz-30000.data-00000-of-00001
    pattern = re.compile("^(model\.npz-(\d+))\.data-00000-of-00001")
    models  = [f for f in files if pattern.match(f)]
    iters   = [int(pattern.match(f).group(2)) for f in models]
    if len(iters) <= modelnum:
        print(" ".join([os.path.join(modeldir, pattern.match(f).group(1)) for f in models]))
    else:
        zipped = zip(iters, models)
        zipped = sorted(zipped, key = lambda t: t[0], reverse=True)
        print(" ".join(
            [os.path.join(modeldir, pattern.match(z[1]).group(1)) for z in zipped[:modelnum]]
        ))


if __name__ == '__main__':
    modeldir = sys.argv[1]
    modelnum = int(sys.argv[2])
    if os.path.isdir(modeldir):
        logger.info("model dir: " + modeldir)
        getmodels(modeldir, modelnum)
    else:
        logger.error("model dir: " + modeldir + " does not exist.")
        sys.exit()
        
