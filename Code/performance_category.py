# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 14:44:20 2023

@author: ruibo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Signal class
class Signal:
    def __init__(self):
        pass
    
    
class SignalAboveAndBelow(Signal):
    
    def __init__(self, series: pd.Series, threshold: np.number):
        
        self.series = series
        self.threshold = threshold
        
    def __call__(self):
        
        # above threshold = 1 | equal or below threshold = 0
        self.output = 1 * (self.series > self.threshold)

        return self.output
    
    
class SignalCrossMovingAverage(Signal):
    
    def __init__(self, series: pd.Series, window: int, expanding_window: bool = False):
        
        self.series = series
        
        assert window > 0
        
        self.window = window
        self.expanding_window = expanding_window
        
        
        
    def __call__(self):
        
        if self.expanding_window:
            self.moving_average = self.series.expanding(self.window).mean().dropna() # expanding average
        else:
            self.moving_average = self.series.rolling(self.window).mean().dropna() # rolling average
        
        # above moving average = 1 | equal or below moving average = 0
        self.output = 1 * (self.series[self.moving_average.index] > self.moving_average)

        return self.output



class SignalMovingZScore(Signal):
    
    def __init__(self, series: pd.Series, window: int, z_score: np.number,
                 expanding_window: bool = False):
        
        self.series = series
        
        assert window > 0
        assert z_score > 0
        
        self.window = window
        self.z_score = z_score
        self.expanding_window = expanding_window
        
        
    def __call__(self):
        
        if self.expanding_window:
            self.moving_average = self.series.expanding(self.window).mean().dropna() # expanding average
            self.moving_std = self.series.expanding(self.window).std().dropna() # expanding standard deviation
            
        else:
            self.moving_average = self.series.rolling(self.window).mean().dropna() # rolling average
            self.moving_std = self.series.rolling(self.window).std().dropna() # rolling standard deviation
            
        self.z_score_series = (self.series[self.moving_average.index] - self.moving_average) / self.moving_std
        
        # below -z: 0 | -z to +z: 1 | above +z: 2
        self.output = pd.Series(index = self.moving_average.index, data = 0) # initialize to 0
        self.output[self.z_score_series > -self.z_score] = 1
        self.output[self.z_score_series > self.z_score] = 2
        
        return self.output



#%%
class PerformanceByCategoty:
    
    def __init__(self, df_performance: pd.Series, df_category: pd.Series, category_names: dict = None):
        
        self.df_performance = df_performance
        
        if category_names is not None:
            # rename category
            df_category = df_category.astype('category').cat.rename_categories(category_names)
        
        self.df_category = df_category
            
        self.df_merge = pd.concat([self.df_performance, self.df_category], axis=1).dropna()
        self.df_merge.columns = ['Performance', 'Category']
        
        self.group_count = self.df_merge.groupby('Category')['Performance'].count()
        self.categories = self.group_count.index
        
    def get_group_mean(self):
        
        self.df_group_mean = self.df_merge.groupby('Category')['Performance'].mean()
        return self.df_group_mean

    
    def get_frequency_above_threshold(self, threshold: np.number = 0):
        
        self.df_frequency = pd.DataFrame(columns = ['Count All', 'Count Above Threshold', 'Frequency'])
        
        self.df_frequency['Count All'] = self.group_count
        self.df_frequency['Count Above Threshold'] = self.df_merge.groupby('Category')['Performance'].apply(lambda x: sum(x > threshold))
        self.df_frequency['Frequency'] = self.df_frequency['Count Above Threshold'] / self.df_frequency['Count All']

        return self.df_frequency
    
    
    def plot_performance_by_category(self, ax = None, figsize: tuple = (15,6), category_colors: list = None, title: str = None,
                                     df_overlay: pd.DataFrame = None, overlay_color: list = None, overlay_style: list = None):
        
        
        df = self.df_merge.copy() # make a copy for plotting
        
        for cat in self.categories:
            df[cat] = None
            df.loc[df['Category'] == cat, cat] = df.loc[df['Category'] == cat, 'Performance']
        
        if df_overlay is not None:
            df = df.join(df_overlay, how='left')
            overlay_fields = df_overlay.columns
        
        df['Date'] = df.index
        df.reset_index(inplace = True)

        # Plotting now
        if ax is None:
            fig, ax = plt.subplots()
            
        df[self.categories].plot(kind='bar', figsize = figsize, color = category_colors, ax = ax)
        
        if df_overlay is not None:
            df[overlay_fields].plot(kind='line', ax = ax, secondary_y = True,
                                    color = overlay_color, style = overlay_style)
        
        x_ticks = list(range(0,len(df), 12))
        x_labels = self.df_merge.index[x_ticks].strftime('%Y')
        
        plt.xticks(ticks=x_ticks, labels=x_labels)
        plt.xlabel('Date')
        
        if title is not None:
            plt.title(title)
        else:
            plt.title('Performance in Different Categories')

        return ax

    def plot_comparing_mean_performance(self, ax = None, figsize: tuple = None, category_colors: list = None, title: str = None):
        
        group_mean = self.get_group_mean()
        
        if ax is None:
            fig, ax = plt.subplots(figsize = figsize)
        
        hbars = ax.bar(group_mean.index, group_mean.values*100, color = category_colors)
        ax.bar_label(hbars, fmt='%.1f%%')
        
        if title is not None:
            plt.title(title)
        else:
            plt.title('Average Performance')
            
        return ax


    def plot_frequency_above_threshold(self, threshold: np.number = 0, ax = None, figsize: tuple = None, category_colors: list = None, title: str = None):
        
        df_frequency = self.get_frequency_above_threshold(threshold)
        
        if ax is None:
            fig, ax = plt.subplots(figsize = figsize)
            
        hbars = ax.bar(df_frequency.index, df_frequency['Frequency'].values*100, color = category_colors)
        ax.bar_label(hbars,fmt='%.1f%%')
        
        x_ticks = list(range(len(df_frequency)))
        x_labels = [_ +'\n Total Count: {} \n Above Threshold: {}'.format(df_frequency.loc[_, 'Count All'], df_frequency.loc[_, 'Count Above Threshold']) for _ in df_frequency.index]
        plt.xticks(ticks=x_ticks, labels=x_labels)
        
        if title is not None:
            plt.title(title)
        else:
            plt.title('Frequency of Above Threshold')
        
        return ax













