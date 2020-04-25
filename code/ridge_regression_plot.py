#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 18:27:48 2020

@author: mai2125
"""


# =============================================================================
# 
# # initialize
# ridge_reg = linear_model.Ridge(alpha=0)
# ridge_reg.fit(X_train, y_train)
# ridge_df = pd.DataFrame({'variable': ps.m_dataset.columns[1:-1], 'estimate': ridge_reg.coef_})
# ridge_train_pred = []
# ridge_test_pred = []
# 
# for alpha in np.arange(0, 200, 1):
#     # training
#     ridge_reg = linear_model.Ridge(alpha=alpha)
#     ridge_reg.fit(X_train, y_train)
#     var_name = 'estimate' + str(alpha)
#     ridge_df[var_name] = ridge_reg.coef_
#     # prediction
#     ridge_train_pred.append(ridge_reg.predict(X_train))
#     ridge_test_pred.append(ridge_reg.predict(X_test))
# 
# # organize dataframe
# ridge_df = ridge_df.set_index("variable").T.reset_index()
# 
# 
# # plot betas by lambda
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.plot(ridge_df.iloc[:, 2], 'r', ridge_df.iloc[:, 5], 'g', ridge_df.iloc[:, 10], 'b',
#         ridge_df.iloc[:, 14], 'm', ridge_df.iloc[:, 20], 'c', ridge_df.iloc[:, 21], 'r-.')
# ax.axhline(y=0, color='black', linestyle='--')
# ax.set_xlabel("Lambda")
# ax.set_ylabel("Beta Estimate")
# ax.set_ylim(-5, 5)
# ax.set_title("Ridge Regression Trace", fontsize=16)
# ax.legend(labels=[coeff_df.index[1], coeff_df.index[4], coeff_df.index[9], coeff_df.index[13],
#           coeff_df.index[19], coeff_df.index[20]])
# ax.grid(True)
# 
# fig2, ax2 = plt.subplots(figsize=(10, 5))
# ax2.plot(ridge_df.iloc[:, 23], 'g', ridge_df.iloc[:, 25], 'b', ridge_df.iloc[:, 26], 'm',
#         ridge_df.iloc[:, 30], 'c', ridge_df.iloc[:, 31], 'r', ridge_df.iloc[:, 33], 'g--')
# ax2.axhline(y=0, color='black', linestyle='--')
# ax2.set_xlabel("Lambda")
# ax2.set_ylabel("Beta Estimate")
# ax2.set_ylim(-5, 5)
# ax2.set_title("Ridge Regression Trace", fontsize=16)
# ax2.legend(labels=[coeff_df.index[22], coeff_df.index[24],
#           coeff_df.index[25], coeff_df.index[29], coeff_df.index[30], coeff_df.index[32]])
# ax2.grid(True)
# 
# fig3, ax3 = plt.subplots(figsize=(10, 5))
# ax3.plot(ridge_df.iloc[:, 35], 'b', ridge_df.iloc[:, 37], 'm', ridge_df.iloc[:, 38], 'c',
#         ridge_df.iloc[:, 40], "r")
# ax3.axhline(y=0, color='black', linestyle='--')
# ax3.set_xlabel("Lambda")
# ax3.set_ylabel("Beta Estimate")
# ax3.set_ylim(-5, 5)
# ax3.set_title("Ridge Regression Trace", fontsize=16)
# ax3.legend(labels=[coeff_df.index[34], coeff_df.index[36], coeff_df.index[37], coeff_df.index[39]])
# ax3.grid(True)
# 
# =============================================================================
        
