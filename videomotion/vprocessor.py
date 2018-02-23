import os
import sys
import getopt
#Import some definitions from modules 
from streams import InputStream, OutputStream
from worker import Worker
from utils import out, err, warn
from visuserver import VisualizationServer
import json


def main(filename, file_dir, arguments):

    # let's print some stuff to debug the provided argument :)
    out()
    out('[Environment]')
    out('- Working dir: ' + os.getcwd())
    out('- Script dir: ' + file_dir)
    out('- Script: ' + filename)
    out('- Arguments: ' + str(arguments))
    out()

    # declaring options
    input_file = ''
    input_folder = ''
    output_folder = ''
    action = ''
    frames = -1
    repetitions = -1
    w = -1
    h = -1
    fps = -1
    force_gray = False

    features = 4
    filter_size = 3
    init_q = 0.1
    step_size = -1

    k = 0.5   # weight of the norm of "q"
    alpha = 0.5  # weight of the norm of the second derivative q^(2)
    beta = 1.0   # weight of the norm of the first derivative q^(1)
    gamma = 2.0   # weight of the mixed-product first-second derivatives q^(2)q^(1)
    theta = 1.0   # exp(theta x t)
    lambda0 = 10.0  # be positive
    lambda1 = 10.0  # sum to 1
    lambdaM = 1.0  # motion constraint
    lambdaC = 2.0  # conditional entropy
    lambdaE = 1.0  # entropy

    eps1 = 100000.0  # bound on the norm of the first derivative q^(1)
    eps2 = 100000.0  # bound on the norm of the second derivative q^(2)
    eps3 = 100000.0  # bound on the norm of the third derivative q^(3)
    zeta = 0.8
    eta = 0.25

    resume = 0
    all_black = 0
    init_fixed = 0

    check_params = True
    grad = False
    day_only = False
    rho = 0.0

    prob_a = -1.0
    prob_b = -1.0

    step_adapt = False
    save_scores_only = False

    # getting options from command line arguments
    if arguments is not None and len(arguments) > 0:
        try:
            opts, args = getopt.getopt(arguments, "", ["resume=", "run=", "pre=", "out=", "res=", "fps=", "frames=",
                                                       "gray=",
                                                       "f=", "m=", "init_q=", "theta=", "alpha=", "beta=",
                                                       "k=", "gamma=", "lambda1=", "lambdaC=", "lambdaM=", "lambda0=",
                                                       "rep=", "eps1=", "eps2=", "eps3=", "zeta=", "eta=",
                                                       "step_size=", "all_black=", "init_fixed=", "check_params=",
                                                       "lambdaE=", "grad=", "rho=", "day_only=", "probA=", "probB=",
                                                       "step_adapt=", "save_scores_only="])
        except getopt.GetoptError:
            usage(filename)
            sys.exit(2)
    else:
        usage(filename)
        sys.exit(2)

    try:
        for opt, arg in opts:
            if opt == '--run':
                if os.path.isdir(arg):
                    input_folder = os.path.abspath(arg)
                    if input_folder.endswith(os.sep):
                        input_folder = input_folder[:-1]
                elif os.path.isfile(arg):
                    input_file = os.path.abspath(arg)
                elif arg == '0':
                    input_file = arg
                else:
                    raise IOError("Cannot open: " + arg)
                action = 'run'
            if opt == '--pre':
                if os.path.isfile(arg):
                    input_file = os.path.abspath(arg)
                else:
                    raise IOError("Cannot open: " + arg)
                action = 'pre-process'
            elif opt == '--out':
                output_folder = os.path.abspath(arg)
                if output_folder.endswith(os.sep):
                    output_folder = output_folder[:-1]
            elif opt == '--res':
                wh = arg.split('x')
                w = int(wh[0])
                h = int(wh[1])
            elif opt == '--gray':
                if int(arg) > 0:
                    force_gray = True
            elif opt == '--fps':
                fps = float(arg)
            elif opt == '--frames':
                frames = int(arg)
            elif opt == '--rep':
                repetitions = int(arg)
            elif opt == '--m':
                features = int(arg)
            elif opt == '--f':
                filter_size = int(arg)
            elif opt == '--init_q':
                init_q = float(arg)
            elif opt == '--theta':
                theta = float(arg)
            elif opt == '--alpha':
                alpha = float(arg)
            elif opt == '--beta':
                beta = float(arg)
            elif opt == '--eps1':
                eps1 = float(arg)
            elif opt == '--eps2':
                eps2 = float(arg)
            elif opt == '--eps3':
                eps3 = float(arg)
            elif opt == '--zeta':
                zeta = float(arg)
            elif opt == '--eta':
                eta = float(arg)
            elif opt == '--k':
                k = float(arg)
            elif opt == '--gamma':
                gamma = float(arg)
            elif opt == '--lambdaC':
                lambdaC = float(arg)
            elif opt == '--lambdaM':
                lambdaM = float(arg)
            elif opt == '--lambda1':
                lambda1 = float(arg)
            elif opt == '--lambda0':
                lambda0 = float(arg)
            elif opt == '--lambdaE':
                lambdaE = float(arg)
            elif opt == '--step_size':
                step_size = float(arg)
            elif opt == '--probA':
                prob_a = float(arg)
            elif opt == '--probB':
                prob_b = float(arg)
            elif opt == '--resume':
                resume = int(arg)
            elif opt == '--all_black':
                all_black = int(arg)
            elif opt == '--init_fixed':
                init_fixed = int(arg)
            elif opt == '--check_params':
                check_params = int(arg) > 0
            elif opt == '--rho':
                rho = float(arg)
            elif opt == '--grad':
                grad = int(arg) > 0
            elif opt == '--day_only':
                day_only = int(arg) > 0
            elif opt == '--step_adapt':
                step_adapt = int(arg) > 0
            elif opt == '--save_scores_only':
                save_scores_only = int(arg) > 0
    except (ValueError, IOError) as e:
        err(e)
        sys.exit(1)

    if len(output_folder) == 0:
        usage(filename)
        sys.exit(2)

    # fixing/forcing options
    if prob_b < prob_a:
        err("Invalid probabilistic range: [" + str(prob_a) + "," + str(prob_b) + "]")
    if repetitions <= 0:
        repetitions = 1
    input_is_video = input_file is not None and len(input_file) > 0
    if not input_file and (w != -1 or h != -1 or fps != -1 or frames != -1):
        warn("Resolution, frame rate, and number of frames are ignored when providing already processed input data.")
        w = -1
        h = -1
        fps = -1

    # printing parsed options
    out('[Parsed Video Options]')
    out('- Action: ' + action)
    out('- Input: ' + (input_file or input_folder))
    out('- Output: ' + output_folder)
    out('- ForceGray: ' + ("yes" if force_gray else "no"))
    out('- Repetitions: ' + str(repetitions))
    out('- Frames: ' + ("empty" if frames <= 0 else str(frames)))
    out()

    # opening the input stream
    input_stream = InputStream(input_file or input_folder, input_is_video)

    # checking option-related errors (given the information acquired from the input stream)
    try:
        if fps != -1 and input_stream.fps < fps:
            raise ValueError("Video FPS: " + input_stream.fps + ", requested FPS: " + fps)
        if fps <= 0:
            fps = input_stream.fps

        if frames != -1 and input_stream.frames < frames:
            raise ValueError('Video frames: ' + str(input_stream.frames) + ', requested frames: ' + str(frames))
        if frames <= 0:
            frames = input_stream.frames

        if w <= 0:
            w = input_stream.w
        if h <= 0:
            h = input_stream.h
        if force_gray:
            c = 1
        else:
            c = input_stream.c

        if step_size < 0.0 < fps:
            step_size = 1.0 / fps
        if step_size < 0.0:
            step_size = 0.01
    except ValueError as e:
        err(e)
        sys.exit(1)

    # printing input stream information
    out('[Input Stream Info]')
    out('- Width: ' + str(input_stream.w))
    out('- Height: ' + str(input_stream.h))
    out('- Channels: ' + str(input_stream.c))
    out('- FPS: ' + ("empty" if input_stream.fps <= 0 else str(input_stream.fps)))
    out('- Frames: ' + ("empty" if input_stream.frames < 0 else str(input_stream.frames)))
    out()

    # setting up the output stream (folder)
    output_stream = OutputStream(output_folder, resume == 0 and (input_is_video or input_folder != output_folder))

    # opening a visualization service (random port)
    if not save_scores_only:
        visualization_server = VisualizationServer(8888, os.path.abspath(file_dir + os.sep + "web"), output_folder)
        out('[Visualization Server]')
        out('- IP: ' + str(visualization_server.ip))
        out('- Port: ' + str(visualization_server.port))
        out('- Data Root: ' + os.path.abspath(output_folder))
        out("- URL: http://" + str(visualization_server.ip) + ":" + str(visualization_server.port))
        out()

    # setting up algorithm options
    options = {'root': os.path.abspath(output_folder),
               'm': features,
               'n': c,
               'f': filter_size,
               'init_q': init_q,
               'theta': theta,
               'alpha': alpha,
               'beta': beta,
               'k': k,
               'gamma': gamma,
               'lambda0': lambda0,
               'lambda1': lambda1,
               'lambdaC': lambdaC,
               'lambdaM': lambdaM,
               'lambdaE': lambdaE,
               'eps1': eps1,
               'eps2': eps2,
               'eps3': eps3,
               'zeta': zeta,
               'eta': eta,
               'step_size': step_size,
               'step_adapt': step_adapt,
               'all_black': all_black,
               'init_fixed': init_fixed,
               'check_params': check_params,
               'rho': rho,
               'day_only': day_only,
               'grad': grad,
               'prob_a': prob_a,
               'prob_b': prob_b,
               'save_scores_only': save_scores_only}

    out('[Algorithm Options]')
    out(json.dumps(options, indent=3))
    out()

    output_stream.save_option("resolution", str(w) + "x" + str(h))
    output_stream.save_option("w", str(w))
    output_stream.save_option("h", str(h))
    output_stream.save_option("n", str(c))
    output_stream.save_option("m", str(features))
    output_stream.save_option("lambda0", str(lambda0))
    output_stream.save_option("lambda1", str(lambda1))
    output_stream.save_option("lambdaC", str(lambdaC))
    output_stream.save_option("lambdaM", str(lambdaM))
    output_stream.save_option("lambdaE", str(lambdaE))
    output_stream.save_option("step_size", str(step_size))
    output_stream.save_option("step_adapt", str(step_adapt))
    output_stream.save_option("f", str(filter_size))
    output_stream.save_option("init_q", str(init_q))
    output_stream.save_option("theta", str(theta))
    output_stream.save_option("alpha", str(alpha))
    output_stream.save_option("beta", str(beta))
    output_stream.save_option("gamma", str(gamma))
    output_stream.save_option("eps1", str(eps1))
    output_stream.save_option("eps2", str(eps2))
    output_stream.save_option("eps3", str(eps3))
    output_stream.save_option("zeta", str(zeta))
    output_stream.save_option("eta", str(eta))
    output_stream.save_option("rho", str(rho))
    output_stream.save_option("prob_a", str(prob_a))
    output_stream.save_option("prob_b", str(prob_b))
    output_stream.save_option("check_params", str(check_params))
    output_stream.save_option("fps", str(fps))
    output_stream.save_option("frames", str(frames * repetitions))
    output_stream.save_option("save_scores_only", str(save_scores_only))
    output_stream.save_option("all_black", str(all_black))
    output_stream.save_option("day_only", str(day_only))
    output_stream.save_option("grad", str(grad))
    output_stream.save_option("init_fixed", str(init_fixed))
    output_stream.save_option("repetitions", str(repetitions))
    if not save_scores_only:
        output_stream.save_option("ip", str(visualization_server.ip))
        output_stream.save_option("port", str(visualization_server.port))
        output_stream.save_option("url", "http://" + str(visualization_server.ip) + ":" + str(visualization_server.port))

    # creating the real worker
    try:
        worker = Worker(input_stream, output_stream, w, h, fps, frames, force_gray, repetitions, options, resume > 0)
    except ValueError as e:
        err(e)
        visualization_server.close()
        sys.exit(1)

    # main loop
    out("Operations are now starting...")
    status = True
    tot_frames = frames * repetitions
    while status:
        try:
            status, frame_time, frame_time_no_save_io, frame_time_no_io = worker.run_step()
        except ValueError as e:
            try:
                worker.close()
            except (ValueError, IOError):
                pass
            err("Error while processing frame " + str(int(worker.steps)+1) + "/" + str(tot_frames))
            err(e)
            if not save_scores_only:
                visualization_server.close()
            sys.exit(1)
        if status:
            out("Processed frame " + str(int(worker.steps)) + "/" + str(tot_frames) +
                " [frame_time: " + "{0:.3f}".format(frame_time) + " (" +
                "{0:.3f}".format(frame_time_no_save_io) + "), avg_fps: " + "{0:.3f}".format(worker.measured_fps) + "]")
    worker.close()
    out("Done!")

    # fixing errors in the frame count procedure
    if worker.steps != tot_frames:
        output_stream.save_option("frames", str(worker.steps))

    # quit the visualization service
    if not save_scores_only:
        visualization_server.close()


def usage(filename):
    out("Usage:")
    out()
    out(filename + ' --run <file/folder/0> --out <folder> [options]')
    out(filename + ' --pre <file/0> --out <folder> [options]')
    out()
    out("where 'file' is video file, and '0' indicates the local web-cam")
    out()
    out("Options can be:")
    out("\t--res <number>x<number>: custom video resolution")
    out("\t--fps <number>: custom video frame rate")
    out("\t--frames <number>: maximum number of frames to process")
    out("\t--m <number>: number of features")
    out("\t--f <number>: filter size (e.g., 3 for 3x3 filters)")
    out()


if __name__ == "__main__":
    main(os.path.basename(sys.argv[0]), os.path.dirname(os.path.abspath(sys.argv[0])), sys.argv[1:])
