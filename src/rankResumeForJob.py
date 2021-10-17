from src.cosineSimRankResume import getCosSimRankResume
from src.clustRankResume import getClustRankResume
import sys, getopt


def main(jobReqDir, resumeDir):
    cosineR = getCosSimRankResume(jobReqDir, resumeDir)
    clustR = getClustRankResume(resumeDir)

    return cosineR.mul(clustR).sort_values(ascending=False)


if __name__ == "__main__":
    jobReqDir = None
    resumeDir = None

    arguments = sys.argv[1:]
    if (len(arguments) != 4):
        print('Usage: ranResumeForJob.py -j <job req dir> -r <resume dir>')
        sys.exit(1)

    try:
        opts, args = getopt.getopt(arguments, 'j:r:')
    except getopt.GetoptError:
        print('Usage: ranResumeForJob.py -j <job req dir> -r <resume dir>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-j':
            jobReqDir = arg
        elif opt == '-r':
            resumeDir = arg
        else:
            print('Usage: ranResumeForJob.py -j <job req dir> -r <resume dir>')
            sys.exit(3)

    resumeRank = main(jobReqDir, resumeDir)
    print(resumeRank)
