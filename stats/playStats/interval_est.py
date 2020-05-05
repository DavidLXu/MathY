from playStats.descriptive_stats import mean,std,variance
from math import sqrt,inf
from scipy.stats import norm,t,chi2,f

# 求均值的置信区间 norm,t
def mean_ci_est(data, alpha, sigma=None):  # confidence interval
    n = len(data)
    sample_mean = mean(data)

    if sigma is None:
        s = std(data)
        se = s/sqrt(n)
        t_value = abs(t.ppf(alpha/2,n-1))
        return sample_mean - se * t_value, sample_mean + se * t_value
    else:
        se = sigma/sqrt(n)
        z_value = abs(norm.ppf(alpha / 2)) # ppf默认下分位点，故使用abs
        return sample_mean - se * z_value, sample_mean + se * z_value

# 求方差的置信区间 chi2
def var_ci_est(data,alpha):
    n = len(data)
    s2 = variance(data)
    chi2_lower_value = chi2.ppf(alpha/2,n-1) # 接受的是左边的面积
    chi2_upper_value = chi2.ppf(1 - alpha/2,n-1)
    return (n-1)*s2/chi2_upper_value,(n-1)*s2/chi2_lower_value

# 方差未知的均值差置信区间 t
def mean_diff_ci_t_est(data1,data2,alpha,equal = True):
    n1 = len(data1)
    n2 = len(data2)
    mean_diff = mean(data1)-mean(data2)

    sample1_var = variance(data1)
    sample2_var = variance(data2)

    if equal:
        sw = sqrt((((n1-1)*sample1_var+(n2-1)*sample2_var))/(n1+n2-2))
        t_value = abs(t.ppf(alpha/2,n1+n2-2))
        return mean_diff - sw * sqrt(1 / n1 + 1 / n2) * t_value, \
               mean_diff + sw * sqrt(1 / n1 + 1 / n2) * t_value
    else:
        df_numerator = (sample1_var/n2+sample2_var/n2)**2                           # 自由度分子
        df_denominator = (sample1_var/n1)**2/(n1-1) + (sample2_var/n2)**2/(n2-1)    # 自由度分母
        df = df_numerator / df_denominator                                          # 自由度
        t_value = abs(t.ppf(alpha/2,df))
        return mean_diff - sqrt(sample1_var / n1 + sample2_var / n2) * t_value, \
               mean_diff + sqrt(sample1_var / n1 + sample2_var / n2) * t_value

# 方差已知求均值差置信区间 norm
def mean_diff_ci_z_est(data1,data2,alpha,sigma1,sigma2):
    n1 = len(data1)
    n2 = len(data2)
    mean_diff = mean(data1) - mean(data2)
    z_value = abs(norm.ppf(alpha/2))
    return mean_diff - sqrt(sigma1 / n1 + sigma2/ n2) * z_value, \
           mean_diff + sqrt(sigma1 / n1 + sigma2 / n2) * z_value

# 均值未知求方差比置信区间 f
def var_ratio_ci_est(data1,data2,alpha):

    n1 = len(data1)
    n2 = len(data2)
    f_lower_value = f.ppf(alpha/2,n1-1,n2-1)
    f_upper_value = f.ppf(1-alpha / 2, n1 - 1, n2 - 1)
    var_ratio = variance(data1) / variance(data2)
    return var_ratio/f_upper_value, var_ratio/f_lower_value

###### 单侧置信区间的实现 #####
# 均值上限置信区间 norm,t
def mean_one_sided_upper_ci_est(data,alpha,sigma = None):
    n = len(data)
    if sigma is None:  # 未知总体方差，使用t分布
        t_value = abs(t.ppf(alpha, n - 1))
        s = std(data)
        return -inf, mean(data) + s / sqrt(n) * t_value

    else:  # 知道总体方差，使用标准正态分布
        z_value = abs(norm.ppf(alpha))
        return -inf, mean(data) + sigma / sqrt(n) * z_value

# 均值下限置信区间 norm,t
def mean_one_sided_lower_ci_est(data,alpha,sigma = None):
    n = len(data)
    if sigma is None:   # 未知总体方差，使用t分布
        t_value = abs(t.ppf(alpha,n-1))
        s = std(data)
        return  mean(data) - s / sqrt(n) * t_value,inf

    else: # 知道总体方差，使用标准正态分布
        z_value = abs(norm.ppf(alpha))
        return mean(data) - sigma/sqrt(n) * z_value,inf

# 方差上限置信区间 chi2
def var_one_sided_upper_ci_est(data,alpha):
    n = len(data)
    s2 = variance(data)
    chi2_lower_value = chi2.ppf(alpha, n - 1)  # 接受的是左边的面积 求上限，分母是下分位点
    return -inf, (n - 1) * s2 / chi2_lower_value

# 方差下限置信区间 chi2
def var_one_sided_lower_ci_est(data,alpha):
    n = len(data)
    s2 = variance(data)
    chi2_upper_value = chi2.ppf(1 - alpha, n - 1) # 接受的是左边的面积 求下限，分母是上分位点
    return (n - 1) * s2 / chi2_upper_value, inf


if __name__ == "__main__":
    salary_18 = [1484, 785, 1598, 1366, 1716, 1020, 1716, 785, 3113, 1601]
    salary_35 = [902, 4508, 3809, 3923, 4726, 2065, 1601, 553, 3345, 2182]
