import motmetrics as mm

def evaluate_tracking(gt_file, res_file):
    gt = mm.io.loadtxt(gt_file, fmt="mot16-02")
    res = mm.io.loadtxt(res_file, fmt="mot16-02")
    mh = mm.metrics.create()
    summary = mh.compute(
        mm.utils.compare_to_groundtruth(gt, res, distth=0.5),
        metrics=['mota', 'idf1', 'precision', 'recall'],
        name='summary'
    )
    print("\nTracking Evaluation Results:")
    print(summary)
