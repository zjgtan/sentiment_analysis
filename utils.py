# coding: utf8

def submit_result(ids, preds, filename):
    fd = open(filename, "w")
    print >> fd, "id,sentiment"
    for i in range(len(ids)):
        print >> fd, "%s,%d" % (ids[i], preds[i])