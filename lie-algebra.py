def SkewFun(a):
    """
    got the corresponded antiSymmetric Matrix of the Lie algebra
    :param a:   Lie algebra
    :return:    antiSymmetric Matrix
    """
    if len(a) == 3:
        A = np.array([[0, -a[2], a[1]],
                      [a[2], 0, -a[0]],
                      [-a[1], a[0], 0]
                      ])
        return A
    if len(a) == 2:
        A = np.array([a[1], -a[0]])
        return A
    exit(-1)