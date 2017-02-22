def score(df):
    # df有三列，ID:学生ID,subsidy_x:实际奖学金金额,subsidy_y:预测奖学金金额
    correct = df[df['subsidy_x'] == df['subsidy_y']]
    s = 0
    for i in [1000, 1500, 2000]:
        r = sum(correct['subsidy_y'] == i)/sum(df['subsidy_x'] == i)
        p = sum(correct['subsidy_y'] == i)/sum(df['subsidy_y'] == i)
        f = r*p*2/(r+p)
        if not np.isnan(f):
            s += (sum(df['subsidy_x'] == i)/df.shape[0])*f
    print(s)