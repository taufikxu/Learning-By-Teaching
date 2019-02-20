from __future__ import division
import numpy as np
import time
import os
import errno
import logging
import coloredlogs


def get_floats(floats_str):
    if not floats_str:
        return []
    return [float(s) for s in floats_str.split(',')]


def get_ints(ints_str):
    if not ints_str:
        return []
    return [int(s) for s in ints_str.split(',')]


# parameters tagging helper function
def param_tag(value):
    if value == 0.0:
        return "00"
    exp = np.floor(np.log10(value))
    leading = ("%e" % value)[0]
    return "%s%d" % (leading, exp)


def setup_logging(args, dirname='output'):
    logger = logging.getLogger('COLOREDLOGS')
    # FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
    FORMAT = '[%(asctime)s] %(message)s'
    DATEFMT = '%H:%M:%S'
    LEVEL_STYLES = dict(
        debug=dict(color='blue'),
        info=dict(color='green'),
        verbose=dict(),
        warning=dict(color='yellow'),
        error=dict(color='red'),
        critical=dict(color='magenta'))
    coloredlogs.install(
        logger=logger,
        level=args.loglv,
        fmt=FORMAT,
        datefmt=DATEFMT,
        level_styles=LEVEL_STYLES)

    # Determine suffix
    suffix = args.dataset if hasattr(args, 'dataset') else ''
    suffix += '-std' + str(args.mog_std) if hasattr(args, 'mog_std') else ''
    suffix += '-scl' + str(args.mog_scale) if hasattr(args, 'mog_scale') else ''
    suffix += '-' if suffix else ''
    suffix += '{{' + (str(args.message)
                      if hasattr(args, 'message') else '') + '}}'
    suffix += '-unrol' + str(args.unrolling_steps) if hasattr(
        args, 'unrolling_steps') else ''
    suffix += '-nl' + str(args.n_layer) if hasattr(args, 'n_layer') else ''
    suffix += '-lr' + str(args.lr) if hasattr(args, 'lr') else ''
    suffix += '-gen_lr' + str(args.gen_lr) if hasattr(args, 'gen_lr') else ''
    suffix += '-est_lr' + str(args.est_lr) if hasattr(args, 'est_lr') else ''
    suffix += '-disc_lr' + str(args.disc_lr) if hasattr(args, 'disc_lr') else ''
    suffix += '-myAdam' if hasattr(args, 'myadam') and args.myadam else ''
    suffix += '-clipgrad' if hasattr(args,
                                     'clip_grad') and args.clip_grad else ''
    suffix += '-wd' + str(args.weight_decay) if hasattr(args,
                                                        'weight_decay') else ''
    suffix += '-bs' + str(args.batch_size) if hasattr(args,
                                                      'batch_size') else ''
    suffix += '-bse' + str(args.batch_size_est) if hasattr(
        args, 'batch_size_est') else ''
    suffix += '-nv' + str(args.n_viz) if hasattr(args, 'n_viz') else ''
    suffix += '-epo' + str(args.max_epoch) if hasattr(args, 'max_epoch') else ''
    suffix += '-lah' + str(args.lookahead) if hasattr(args, 'lookahead') else ''
    suffix += '-seed' + str(args.seed) if hasattr(args, 'seed') else ''

    # Determine prefix
    prefix = time.strftime("%Y-%m-%d--%H-%M")

    prefix_counter = 0
    dirname = dirname + "/%s.%s" % (prefix, suffix)
    while True:
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise e
            prefix_counter += 1
            dirname = "output/%s+%d.%s" % (prefix, prefix_counter, suffix)
        else:
            break

    formatter = logging.Formatter(FORMAT, DATEFMT)
    logger_fname = os.path.join(dirname, "logfile.txt")
    fh = logging.FileHandler(logger_fname)
    # fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger, dirname
