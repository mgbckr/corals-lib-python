import pytest


def test_issue1():

    data = """
Name	ERR4165185	ERR4165186	ERR4165187	ERR4165188	ERR4165189	ERR4165190	ERR4165191	ERR4165192	ERR4165193	ERR4165194
Scp1_US851008_k31_TRINITY_DN18756_c0_g1_i4	0	0	0	0	0	0	0	0	0	0
Scp1_US851008_k25_TRINITY_DN10094_c0_g1_i2	0	5.94091	5.58655	5.81978	5.39608	8.5646	6.63334	6.29947	4.76114	2.87519
Scp1_US851008_k25_TRINITY_DN7610_c1_g1_i24	0	0.584322	0.361902	0.372344	0.372583	0	0.16313	0	0.874301	0.567489
Scp1_US851008_k25_TRINITY_DN3138_c0_g1_i2	0	3.66967	2.61626	4.58642	1.00118	7.74655	6.50818	3.50205	2.80706	3.18808
Scp1_US851008_k25_TRINITY_DN66949_c0_g1_i1	0	0.55958	0.508452	0.188733	0	2.05569	0.622747	0.18403	1.22889	0.525269
Scp1_US851008_k25_TRINITY_DN42729_c0_g1_i3	0	NaN	0	0	0	0	0	0	4.98475	0
Scp1_US851008_k25_TRINITY_DN5537_c0_g1_i1	0	0	0	0	0.068946	0.997404	0.394103	0.13994	0.375641	3.50364
Scp1_US851008_k31_TRINITY_DN9195_c0_g2_i2	0	1.38316	0.785248	2.0806	1.17822	0	0	0	1.1218	5.1715
Scp1_US851008_k31_TRINITY_DN9068_c0_g1_i22	0	0	0	0	0	0	0.164973	0.296276	0	2.88606
"""
    
    from corals.threads import set_threads_for_external_libraries
    set_threads_for_external_libraries(n_threads=1)

    import pandas as pd
    from io import StringIO
    from corals.correlation.topk.default import cor_topk

    df = pd.read_csv(StringIO(data), sep='\t')
    df.set_index('Name', inplace=True)
    df_transposed = df.T

    with pytest.raises(ValueError) as e:
        cor_topk(df_transposed, k=0.001, correlation_type="spearman", n_jobs=4)
    assert e.match(r"^Zero variance in X\. Please remove\. Indices: \[0\]$")


if __name__ == '__main__':
    test_issue1()