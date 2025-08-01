#region imports
from AlgorithmImports import *
import pandas as pd
import numpy as np
import decimal as dc
import datetime as dt
from enumerators import (SignalOnSameDirection, SignalOnOppositeDirection,
                        SignalNeutral, TrainingFrequency,
                        SignalCheckFrequency, SIGNAL_LONG, SIGNAL_SHORT,
                        SIGNAL_NONE, SIGNAL_LONG_STRONG, SIGNAL_SHORT_STRONG)
from signal_handler import SignalHandler
#endregion

#region constants
INVALID = -1234
MAX_RW_SIZE = 6000

FEATURE_O = 1
FEATURE_A1 = 2
FEATURE_A2 = 3
FEATURE_B1 = 4
FEATURE_B2 = 5
FEATURE_C1 = 6
FEATURE_C2 = 7
#endregion

class AssetData(object):

    def __init__(self, algorithm:QCAlgorithm, symbol, resolution,
            stops_target, stops_target_strong, strong_signal_multiplier,
            features_size,
            bar_result, min_train_size,
            max_train_size,
            confidence, category_b, category_c,
            training_frequency, signal_frequency,
            ai_model,
            signal_same_direction, signal_opp_direction, signal_neutral,
            max_weight):

        self.symbol = symbol.Symbol
        self._symbol = symbol

        self.algorithm = algorithm
        self.resolution = resolution

        # Risk Management
        self.max_weight = max_weight
        self.signal_same_direction = signal_same_direction
        self.signal_opp_direction = signal_opp_direction
        self.signal_neutral = signal_neutral

        # AI Model
        self.ai_model = ai_model

        # AI Training
        self.features_size = features_size
        self.bar_result = bar_result
        self.stoptarget = stops_target
        self.stoptarget_strong = stops_target_strong
        self.profittarget = stops_target
        self.profittarget_strong = stops_target_strong
        self.strong_signal_multiplier = strong_signal_multiplier
        self.minimum_train_size = min_train_size
        self.maximum_train_size = max_train_size
        self.confidence = confidence
        self.training_frequency = training_frequency

        # AI Signal
        self.indicator_category_b = category_b
        self.indicator_category_c = category_c
        self.signal_frequency = signal_frequency

        # Indicator Parameters
        # RSI
        self.rsi_period = 3
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.rsi_extreme_oversold = 15
        self.rsi_extreme_overbought = 85
        # BB
        self.bb_period = 3
        self.bb_dev = 2.0
        # MA Fast
        self.ma_fast_period = 8
        # MA Slow
        self.ma_slow_period = 50
        # PSAR
        self.sar_start = 0.02
        self.sar_step = 0.02
        self.sar_max = 0.2
        # ATR
        self.atr_period = 6

        # Create indicators
        self.rsi = RelativeStrengthIndex(self.rsi_period)
        self.bb = BollingerBands(self.bb_period, self.bb_dev)
        self.ma_fast = ExponentialMovingAverage(self.ma_fast_period)
        self.ma_slow = ExponentialMovingAverage(self.ma_slow_period)
        self.psar = ParabolicStopAndReverse(self.sar_start, self.sar_step, self.sar_max)
        self.atr = AverageTrueRange(self.atr_period, MovingAverageType.SIMPLE)

        # Rolling Windows
        self.rwClose = RollingWindow[float](MAX_RW_SIZE)
        self.rwRSI = RollingWindow[float](MAX_RW_SIZE)
        self.rwBBLower = RollingWindow[float](MAX_RW_SIZE)
        self.rwBBUpper = RollingWindow[float](MAX_RW_SIZE)
        self.rwMAFast = RollingWindow[float](MAX_RW_SIZE)
        self.rwMASlow = RollingWindow[float](MAX_RW_SIZE)
        self.rwPSar= RollingWindow[float](MAX_RW_SIZE)
        self.rwBar = RollingWindow[str](MAX_RW_SIZE)
        self.rwAtr = RollingWindow[float](MAX_RW_SIZE)
        self.rwTr = RollingWindow[float](MAX_RW_SIZE)
        self.rwTime = RollingWindow[str](MAX_RW_SIZE)

        # AI
        self.model = None
        self.learn_model = None

        # AI Training
        if self.training_frequency == TrainingFrequency.MONTHLY:
            self.event_training = self.algorithm.schedule.on(self.algorithm.date_rules.month_start(),
                    self.algorithm.time_rules.after_market_open(self.symbol, 60),
                    self.ScheduleTrain)

        elif self.training_frequency == TrainingFrequency.WEEKLY:
            self.event_training = self.algorithm.schedule.on(self.algorithm.date_rules.week_start(),
                    self.algorithm.time_rules.after_market_open(self.symbol, 60),
                    self.ScheduleTrain)

        elif self.training_frequency == TrainingFrequency.DAILY:
            self.event_training = self.algorithm.schedule.on(self.algorithm.date_rules.every_day(),
                    self.algorithm.time_rules.after_market_open(self.symbol, 60),
                    self.ScheduleTrain)

        # AI Predict
        if self.signal_frequency == SignalCheckFrequency.AFTER_MARKET_OPEN or self.signal_frequency == SignalCheckFrequency.TWICE_A_DAY:
            self.event_predict = self.algorithm.schedule.on(
                    self.algorithm.date_rules.every_day(),
                    self.algorithm.time_rules.after_market_open(self.symbol, 60),
                    self.ScheduleAction)

        if self.signal_frequency == SignalCheckFrequency.BEFORE_MARKET_CLOSE or self.signal_frequency == SignalCheckFrequency.TWICE_A_DAY:

            self.event_predict = self.algorithm.schedule.on(
                    self.algorithm.date_rules.every_day(),
                    self.algorithm.time_rules.before_market_close(self.symbol, 60),
                    self.ScheduleAction)

    def train(self):
        if not self.isReady():
            return

        traindata, results= self.getPastFeatures()
        if traindata is None:
            return

        features = traindata.values.astype(np.float32)
        target = results.values.astype(np.float32)

        self.learn_model = self.ai_model

        self.model = self.learn_model.fit(features, target)

    def Stop(self):
        self.algorithm.log('Removing from Scheduled Events: %s' % self.symbol)
        self.algorithm.schedule.remove(self.event_training)
        self.algorithm.schedule.remove(self.event_predict)

        if self.algorithm.portfolio[self.symbol].invested:
            self.algorithm.log('Closing positions: %s' % self.symbol)
            self.algorithm.liquidate(self.symbol)

    def ScheduleTrain(self):
        history = self.algorithm.history[TradeBar](self.symbol,
                    self.maximum_train_size+self.bar_result,
                    resolution=self.resolution
                    )

        self.reload(history)
        self.train()

    def ScheduleAction(self):
        signal = self.signal()

        if signal == INVALID:
            return

        confidence = 1 - (self.model.loss_/self.confidence)

        current_weight = self.algorithm.portfolio[self.symbol].holdings_value / self.algorithm.portfolio.total_portfolio_value
        if current_weight < 0: current_weight *= -1
        weight = confidence + current_weight

        signal_str = 'Invalid'
        if signal == SIGNAL_NONE:
            signal_str = 'None'
        elif signal == SIGNAL_LONG:
            signal_str = 'Buy'
        elif signal == SIGNAL_SHORT:
            signal_str = 'Sell'
        elif signal == SIGNAL_LONG_STRONG:
            signal_str = 'Strong Buy'
        elif signal == SIGNAL_SHORT_STRONG:
            signal_str = 'Strong Sell'

        if signal == SIGNAL_LONG_STRONG or signal == SIGNAL_SHORT_STRONG:
            weight *= self.strong_signal_multiplier

        if weight > self.max_weight:
            weight = self.max_weight

        self.algorithm.debug(f'{self.symbol} | Signal: {signal_str} | Model Loss: {self.model.loss_:.4f} | Confidence: {confidence:.4f} | Current Weight: {current_weight:.4f} | New Weight: {weight:.4f}')

        holdings = self.algorithm.portfolio[self.symbol].quantity
        unpnl = self.algorithm.portfolio[self.symbol].unrealized_profit

        str_debug1 = ''.join([f'{self.symbol} | {signal_str} | Holdings: {holdings} | UnPnL: {unpnl:.2f} | ',
                             f'Signal Confidence: {confidence:.2f} | Signal Weight: {weight:.2f} | Current: {current_weight:.4f} | ',
                             f'Model Loss: {self.model.loss_:.4f} | Training Samples: {self.model.t_} | Model Iterations: {self.model.n_iter_} | '
                            ])

        str_debug2 = ''.join([
                             f'Classes: {self.model.classes_} | Exit Layer Activation : {self.model.out_activation_} | ',
                             f'Shape First Layer Weights: {self.model.coefs_[0].shape} | Shape First Layer Biases: {self.model.intercepts_[0].shape} | '
                            ])

        if hasattr(self.model, 'loss_curve_'):
            str_debug2 = ''.join([str_debug2, 
                                 f'Loss Curve Lenght: {len(self.model.loss_curve_)} | '
                                 f'Loss Curve Last Values:{self.model.loss_curve_[-3]:.3f} ,  {self.model.loss_curve_[-2]:.3f} , {self.model.loss_curve_[-1]:.3f} | ', 
                                ])

        if round(holdings) != 0 and signal != SIGNAL_NONE:
            self.algorithm.debug(str_debug1)
            #self.algorithm.debug(str_debug2)

        if self.model.loss_ >= self.confidence or self.model.loss_ <0:
            return

        SignalHandler(self.algorithm, self.symbol, signal, confidence, weight,
                      self.signal_same_direction, self.signal_opp_direction, self.signal_neutral, 
                      self.resolution, self.bar_result) 

    def reload(self, history):
        # Reset indicators
        self.rsi = RelativeStrengthIndex(self.rsi_period)
        self.bb = BollingerBands(self.bb_period, self.bb_dev)
        self.ma_fast = ExponentialMovingAverage(self.ma_fast_period) 
        self.ma_slow = ExponentialMovingAverage(self.ma_slow_period)
        self.psar = ParabolicStopAndReverse(self.sar_start, self.sar_step, self.sar_max)
        self.atr = AverageTrueRange(self.atr_period, MovingAverageType.SIMPLE)

        # Clear rolling windows
        self.rwClose.reset()
        self.rwRSI.reset()
        self.rwBBLower.reset()
        self.rwBBUpper.reset()
        self.rwMAFast.reset()
        self.rwMASlow.reset()
        self.rwPSar.reset()
        self.rwAtr.reset()
        self.rwTr.reset()
        self.rwTime.reset()

        for bar in history:
            self.update(bar)

    def isReady(self):
        return (self.rwClose.count > self.minimum_train_size + self.bar_result)

    def update(self, bar:TradeBar):
        self.rsi.Update(bar.time, bar.close)
        self.bb.Update(bar.time, bar.close)
        self.ma_fast.Update(bar.time, bar.close)
        self.ma_slow.Update(bar.time, bar.close)
        self.psar.Update(bar)
        self.atr.Update(bar)

        if self.ma_slow.IsReady:# highest period
            self.rwClose.add(bar.close)
            self.rwRSI.add(self.rsi.Current.Value)
            self.rwBBLower.add(self.bb.UpperBand.Current.Value)
            self.rwBBUpper.add(self.bb.LowerBand.Current.Value)
            self.rwMAFast.add(self.ma_fast.Current.Value)
            self.rwMASlow.add(self.ma_slow.Current.Value)
            self.rwPSar.add(self.psar.Current.Value)
            self.rwBar.add(str(bar.time))
            self.rwAtr.add(self.atr.Current.Value)
            self.rwTr.add(self.atr.TrueRange.Current.Value)
            self.rwTime.add(bar.time.strftime('%Y-%m-%d %H:%M'))

    def signal(self):
        if not self.isReady():
            return INVALID

        if self.learn_model is None:
            return INVALID

        features = self.getCurrentFeatures()
        if features is None:
            return INVALID

        return self.learn_model.predict(features)[0]

    def getPastFeatures(self):
        featuresdf = pd.DataFrame()
        resultdf = pd.DataFrame()

        if self.rwClose.count < self.minimum_train_size or not self.isReady():
            return None, None

        # Prevent the loop from being made in a larger number of candles, due to the BAR_RESULT
        max_train = self.maximum_train_size
        if max_train > self.rwClose.count - self.bar_result:
            max_train = self.rwClose.count - self.bar_result -1

        data = dict()
        for j in range(max_train, 1+self.features_size+self.bar_result, -self.features_size-1):
            for i in range(self.features_size+1, 0, -1):

                data['BB'+str(i)] = self.getBBFeature(j-i)
                data['RSI'+str(i)] = self.getRsiFeature(j-i)
                data['MAFast'+str(i)] = self.getMAFastFeature(j-i)
                data['MASlow'+str(i)] = self.getMASlowFeature(j-i)
                data['PSAR'+str(i)] = self.getPSarFeature(j-i)
                data['ATR'+str(i)] = self.getAtrFeature(j-i)
                data['TR'+str(i)] = self.getTrFeature(j-i)

            resultdf = pd.concat([resultdf, 
                    pd.DataFrame(data=[self.getResultBar(j-self.features_size-2)])])

            featuresdf = pd.concat([featuresdf, pd.DataFrame(data, index=[featuresdf.shape[0]])])

        return (featuresdf, resultdf)

    def getCurrentFeatures(self):
        if not self.isReady():
            return None

        df = pd.DataFrame()
        data = dict()
        for i in range(self.features_size+1, 0, -1):
            data['BB'+str(i)] = self.getBBFeature(i)
            data['RSI'+str(i)] = self.getRsiFeature(i)
            data['MAFast'+str(i)] = self.getMAFastFeature(i)
            data['MASlow'+str(i)] = self.getMASlowFeature(i)
            data['PSAR'+str(i)] = self.getPSarFeature(i)
            data['ATR'+str(i)] = self.getAtrFeature(i)
            data['TR'+str(i)] = self.getTrFeature(i)

        df = pd.DataFrame(data, index=[df.shape[0]])
        return df.astype(np.float32)

    def getResultBar(self, n):
        sl = dc.Decimal(self.rwClose[n]) * (dc.Decimal(1) - dc.Decimal(self.stoptarget))
        tp = dc.Decimal(self.rwClose[n]) * (dc.Decimal(1) + dc.Decimal(self.profittarget))
        sl_strong = dc.Decimal(self.rwClose[n]) * (dc.Decimal(1) - dc.Decimal(self.stoptarget_strong))
        tp_strong = dc.Decimal(self.rwClose[n]) * (dc.Decimal(1) + dc.Decimal(self.profittarget_strong))

        for i in range(0, self.bar_result, 1):

            if self.rwClose[n-i] >= tp_strong:
                return(SIGNAL_LONG_STRONG)

            elif self.rwClose[n-i] <= sl_strong:
                return(SIGNAL_SHORT_STRONG)

            elif self.rwClose[n-i] >= tp:
                return(SIGNAL_LONG)

            elif self.rwClose[n-i] <= sl:
                return(SIGNAL_SHORT)

        return(SIGNAL_NONE)

    def getBBFeature(self, n):
        if self.rwClose[n] > self.rwBBUpper[n] * (1 + self.indicator_category_c):
            return(FEATURE_C1)

        elif self.rwClose[n] <  self.rwBBLower[n] * (1 - self.indicator_category_c):
            return(FEATURE_C2)

        elif self.rwClose[n] > self.rwBBUpper[n] * (1 + self.indicator_category_b):
            return(FEATURE_B1)

        elif self.rwClose[n] <  self.rwBBLower[n] * (1 - self.indicator_category_b):
            return(FEATURE_B2)

        elif self.rwClose[n] > self.rwBBUpper[n]:
            return(FEATURE_A1)

        elif self.rwClose[n] < self.rwBBLower[n]:
            return(FEATURE_A2)

        else:
            return(FEATURE_O)

    def getRsiFeature(self, n):
        if self.rwRSI[n] > self.rsi_extreme_oversold:
            return(FEATURE_C1)

        elif self.rwRSI[n] < self.rsi_extreme_overbought:
            return(FEATURE_C2)

        elif self.rwRSI[n] > self.rsi_oversold:
            return(FEATURE_B1)

        elif self.rwRSI[n] < self.rsi_overbought:
            return(FEATURE_B2)

        elif self.rwRSI[n] > 50:
            return(FEATURE_A1)

        elif self.rwRSI[n] < 50:
            return(FEATURE_A2)

        else:
            return(FEATURE_O)

    def getMAFastFeature(self, n):
        if self.rwMAFast[n] > self.rwClose[n] * (1 + self.indicator_category_c):
            return(FEATURE_C1)

        elif self.rwMAFast[n] < self.rwClose[n] * (1 - self.indicator_category_c):
            return(FEATURE_C2)

        elif self.rwMAFast[n] > self.rwClose[n] * (1 + self.indicator_category_b):
            return(FEATURE_B1)

        elif self.rwMAFast[n] < self.rwClose[n] * (1 - self.indicator_category_b):
            return(FEATURE_B2)

        elif self.rwMAFast[n] > self.rwClose[n]:
            return(FEATURE_A1)

        elif self.rwMAFast[n] < self.rwClose[n]:
            return(FEATURE_A2)

        else:
            return(FEATURE_O)
    
    def getMASlowFeature(self, n):
        if self.rwMASlow[n] > self.rwClose[n] * (1 + self.indicator_category_c):
            return(FEATURE_C1)

        elif self.rwMASlow[n] < self.rwClose[n] * (1 - self.indicator_category_c):
            return(FEATURE_C2)

        elif self.rwMASlow[n] > self.rwClose[n] * (1 + self.indicator_category_b):
            return(FEATURE_B1)

        elif self.rwMASlow[n] < self.rwClose[n] * (1 - self.indicator_category_b):
            return(FEATURE_B2)

        elif self.rwMASlow[n] > self.rwClose[n]:
            return(FEATURE_A1)

        elif self.rwMASlow[n] < self.rwClose[n]:
            return(FEATURE_A2)

        else:
            return(FEATURE_O)

    def getAtrFeature(self, n):
        if self.rwAtr[n] > self.rwAtr[n+1] * (1 + self.indicator_category_c):
            return(FEATURE_C1)

        elif self.rwAtr[n] < self.rwAtr[n+1] * (1 - self.indicator_category_c):
            return(FEATURE_C2)

        elif self.rwAtr[n] > self.rwAtr[n+1] * (1 + self.indicator_category_b):
            return(FEATURE_B1)

        elif self.rwAtr[n] < self.rwAtr[n+1] * (1 - self.indicator_category_b):
            return(FEATURE_B2)

        elif self.rwAtr[n] > self.rwAtr[n+1]:
            return(FEATURE_A1)

        elif self.rwAtr[n] < self.rwAtr[n+1]:
            return(FEATURE_A2)

        else:
            return(FEATURE_O)

    def getTrFeature(self, n):
        if self.rwTr[n] > self.rwTr[n+1] * (1 + self.indicator_category_c):
            return(FEATURE_C1)

        elif self.rwTr[n] < self.rwTr[n+1] * (1 - self.indicator_category_c):
            return(FEATURE_C2)

        elif self.rwTr[n] > self.rwTr[n+1] * (1 + self.indicator_category_b):
            return(FEATURE_B1)
            
        elif self.rwTr[n] < self.rwTr[n+1] * (1 - self.indicator_category_b):
            return(FEATURE_B2)

        elif self.rwTr[n] > self.rwTr[n+1]:
            return(FEATURE_A1)

        elif self.rwTr[n] < self.rwTr[n+1]:
            return(FEATURE_A2)

        else:
            return(FEATURE_O)

    def getPSarFeature(self, n):
        if self.rwPSar[n] > self.rwClose[n] * (1 + self.indicator_category_c):
            return(FEATURE_C1)

        elif self.rwPSar[n] < self.rwClose[n] * (1 - self.indicator_category_c):
            return(FEATURE_C2)

        elif self.rwPSar[n] > self.rwClose[n] * (1 + self.indicator_category_b):
            return(FEATURE_B1)

        elif self.rwPSar[n] < self.rwClose[n] * (1 - self.indicator_category_b):
            return(FEATURE_B2)

        elif self.rwPSar[n] > self.rwClose[n]:
            return(FEATURE_A1)

        elif self.rwPSar[n] < self.rwClose[n]:
            return(FEATURE_A2)

        else:
            return(FEATURE_O)
