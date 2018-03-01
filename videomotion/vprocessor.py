import os
import sys
import getopt
#Import some definitions from modules 
from streams import InputStream, OutputStream
from worker import Worker
from utils import out, err, warn
from visuserver import VisualizationServer
import json

#This is the main program
# test
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
    visualization_port = 0

    features = 4  # number of features
    filter_size = 3  # "edge" size of a filter, i.e., the area is filter_size x filter_size
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
    rho = 0.0
    init_q = 0.1
    prob_a = -1.0  # eps-insensitive probabilistic constraint
    prob_b = -1.0  # eps-insensitive probabilistic constraint

    step_adapt = False
    step_size = -1

    resume = 0
    all_black = 0
    init_fixed = 0
    grad = False
    day_only = False
    save_scores_only = False
    check_params = True



    #Now we start to process the command line.
    
    # Getting options from command line arguments
    accepted_options = ["port=", "resume=", "run=", "out=", "res=", "fps=", "frames=",
                        "gray=", "f=", "m=", "init_q=", "theta=", "alpha=", "beta=",
                        "k=", "gamma=", "lambda0=", "lambda1=", "lambdaC=", "lambdaE=", "lambdaM=",
                        "rep=", "eps1=", "eps2=", "eps3=", "zeta=", "eta=",
                        "step_size=", "all_black=", "init_fixed=", "check_params=",
                        "grad=", "rho=", "day_only=", "probA=", "probB=",
                        "step_adapt=", "save_scores_only="]
    description = ["port of the visualization service", "resume an experiment (binary flag)",
                   "none", "none", "input resolution (example: 240x120)",
                   "frames per second", "maximum number of frames to consider", "force gray scale (binary flag)",
                   "edge of a filter (example for a 3x3 filter: 3)", "number of features",
                   "maximum absolute value of initial components of q",
                   "exp(theta x t)", "weight of the norm of the second derivative of q",
                   "weight of the norm of the first derivative of q", "weight of the norm of q",
                   "weight of the mixed-product first-second derivatives", "weight of the positivity constraint",
                   "weight of the sum-to-1 constraint", "weight of the conditional entropy",
                   "weight of the entropy", "weight of the motion constraint",
                   "number of video repetitions",
                   "bound on the norm of the first derivative of q",
                   "bound on the norm of the second derivative of q",
                   "bound on the norm of the third derivative of q",
                   "norm-bounds scaling factor during night",
                   "update factor for parameter rho",
                   "size of the step in the Euler's method (or learning rate in the gradient-based update)",
                   "force input to be all black",
                   "initialize all the components of q to init_q",
                   "check validity of the alpha, beta, gamma, theta, and k parameters (binary flag)",
                   "switch to online stochastic gradient (binary flag)",
                   "blurring factor",
                   "force the system to work only in day-mode",
                   "lower bound of the bound-based probabilistic constraint",
                   "upper bound of the bound-based probabilistic constraint",
                   "use adaptive step_size (binary flag)",
                   "do not save output data, with the exception of the scalar scores (binary flag)"]

    #QQQ:Why does we are using "is not None"?
    if arguments is not None and len(arguments) > 0:
        # If there are arguments parse the command line, if something unexpected comes up,
        #launch the "usage" function with approptiate usage message
        try:
            opts, args = getopt.getopt(arguments, "", accepted_options)
        except getopt.GetoptError:
            usage(filename, accepted_options, description)
            sys.exit(2)
    else:
        usage(filename, accepted_options, description)
        sys.exit(2)

    #Now for the real processing of the command line we match the specified arguments with the corresponding variables.
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
            elif opt == '--port':
                visualization_port = int(arg)
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
        usage(filename, accepted_options, description)
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
    #QQQ: What is InputStream? Where this comes from?
    #InputStream is a class defined in streams.py
    input_stream = InputStream(input_file or input_folder, input_is_video)

    # checking option-related errors (given the information acquired from the input stream)
    #QQQ: What are this checks/ assignments? A: Adjust some variables.
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
    tot_frames = frames * repetitions

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
    visualization_server = None
    if not save_scores_only:
        visualization_server = VisualizationServer(visualization_port,
                                                   os.path.abspath(file_dir + os.sep + "web"), output_folder)
        out('[Visualization Server]')
        out('- IP: ' + str(visualization_server.ip))
        out('- Port: ' + str(visualization_server.port))
        out('- Data Root: ' + os.path.abspath(output_folder))
        out("- URL: http://" + str(visualization_server.ip) + ":" + str(visualization_server.port))
        out()

    # setting up algorithm options
    #Create a dictionary that consists of all the useful informations that we want to pass to the routines we
    #write to solve the problem.
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

    #QQQ: What are these options? A: save the specified options.
    output_stream.save_option("resolution", str(w) + "x" + str(h))
    output_stream.save_option("w", str(w))
    output_stream.save_option("h", str(h))
    output_stream.save_option("fps", str(fps))
    output_stream.save_option("frames", str(tot_frames))
    output_stream.save_option("repetitions", str(repetitions))

    for k, v in options.items():
        output_stream.save_option(k, str(v))

    if not save_scores_only:
        output_stream.save_option("ip", str(visualization_server.ip))
        output_stream.save_option("port", str(visualization_server.port))
        output_stream.save_option("url", "http://" + str(visualization_server.ip) + ":"
                                  + str(visualization_server.port))

    # creating the real worker
    try:
        worker = Worker(input_stream, output_stream, w, h, fps, frames, force_gray, repetitions, options, resume > 0)
    except ValueError as e:
        err(e)
        if not save_scores_only:
            visualization_server.close()
        sys.exit(1)

    # main loop
    out("Operations are now starting...")
    status = True
    while status:
        try:
            #Process a frame at a time.
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


def usage(filename, accepted_options, description):
    out("Usage:")
    out()
    out(filename + ' --run <file/folder/0> --out <folder> [options]')
    out()
    out("where 'file' is video file, and '0' indicates the local web-cam")
    out()
    out("Options can be (some descriptions will be added soon):")
    i = 0
    for k in accepted_options:
        if i < len(description):
            if description[i] != "none":
                tab = (18 - len(k[:-1])) * " "
                out("   --" + k[:-1] + ":" + tab + description[i])
        else:
            out("   --" + k[:-1])
        i = i + 1
    out()


#Entry point.
if __name__ == "__main__":
    main(os.path.basename(sys.argv[0]), os.path.dirname(os.path.abspath(sys.argv[0])), sys.argv[1:])
#QQQ: What is this?
