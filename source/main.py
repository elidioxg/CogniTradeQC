#region imports
from AlgorithmImports import *

import datetime as dt

from asset_data import AssetData

from enumerators import (TrainingFrequency, SignalCheckFrequency, 
                        SignalOnOppositeDirection, SignalNeutral, 
                        SignalOnSameDirection, PortfolioConstructor,
                        OrderExecutionType)

from ai_models import GetModelMLPClassifier
#endregion

BENCHMARK = "QTEC"
RESOLUTION = Resolution.HOUR

# Risk Management
TRAILING_STOP_RISK = 0 # (0 = disabled)
MAX_UNPNL_PROFIT = 0 # Take Profit (0 = disabled)
MAX_DRAWDOWN = 0 # Stop Loss (0 = disabled)
MAX_DRAWDOWN_PORTFOLIO = 0 # (0 = disabled)

# Portfolio Construction
PORTFOLIO_CONSTRUCTOR = PortfolioConstructor.INSIGHT_WEIGHTING
MAX_WEIGHT = 0.5
LEVERAGE = 1

# Orders
ORDER_EXECUTION = OrderExecutionType.IMMEDIATE

# Training
TRAINING_FREQUENCY = TrainingFrequency.WEEKLY
STOPS_CHECK_TARGET = 2 # How many bars to check TP and SL target
STOPS_TARGET = 0.005 # TP and SL target to train the model

# Training Strong Signals
STOPS_TARGET_STRONG = 0.01 # TP and SL target to train the model
STRONG_SIGNAL_WEIGHT_MULTIPLIER = 1.5

# Signal
SIGNAL_FREQUENCY = SignalCheckFrequency.AFTER_MARKET_OPEN
CONFIDENCE_CUTOFF = 2

# Indicator Category Model Classification
INDICATOR_CATEGORY_B = 0.01
INDICATOR_CATEGORY_C = 0.02

# Signals with Opened Positions from Same Symbol
SIGNAL_SAME_DIRECTION = SignalOnSameDirection.EMMIT_IF_STRONG_SIGNAL_AND_NOT_PROFITABLE
SIGNAL_OPPOSITE_DIRECTION = SignalOnOppositeDirection.EMMIT_IF_STRONG_SIGNAL
SIGNAL_NEUTRAL = SignalNeutral.CLOSE_IF_PROFITABLE

# AI Model
FEATURES_SIZE = 3
HIDDEN_LAYERS = (20)
# Training Data Size
MINIMUM_TRAIN_SIZE = 3800
MAXIMUM_TRAIN_SIZE = 3900

class CogniTrade(QCAlgorithm):

    def Initialize(self):

        # Backtesting parameters
        self.set_start_date(2024, 1, 1)
        self.set_end_date(2024, 3, 1)
        self.set_cash(1000000)

        # General Configuration
        self.set_warm_up(dt.timedelta(days=1))
        self.set_benchmark(BENCHMARK)
        self.set_brokerage_model(BrokerageName.ALPHA_STREAMS)

        # Universe Selection)
        self.set_universe_selection(TechnologyETFUniverse())
        self.universe_settings.resolution = Resolution.DAILY
        self.universe_settings.minimum_time_in_universe = dt.timedelta(days=7)
        self.universe_settings.leverage = LEVERAGE
        
        # Risk Management
        if TRAILING_STOP_RISK > 0:
            self.add_risk_management(TrailingStopRiskManagementModel(TRAILING_STOP_RISK))
        if MAX_UNPNL_PROFIT > 0:
            self.add_risk_management(MaximumUnrealizedProfitPercentPerSecurity(MAX_UNPNL_PROFIT))
        if MAX_DRAWDOWN > 0:
            self.add_risk_management(MaximumDrawdownPercentPerSecurity(MAX_DRAWDOWN))
        if MAX_DRAWDOWN_PORTFOLIO > 0:
            self.add_risk_management(MaximumDrawdownPercentPortfolio(MAX_DRAWDOWN_PORTFOLIO, True))

        # Internal vars
        self.symbols_universe = {}

        # Portfolio Construction
        if PORTFOLIO_CONSTRUCTOR == PortfolioConstructor.ACCUMULATIVE_INSIGHT:
            self.set_portfolio_construction(
                AccumulativeInsightPortfolioConstructionModel(
                Resolution.DAILY, PortfolioBias.LONG_SHORT,
                MAX_WEIGHT
                ))

        elif PORTFOLIO_CONSTRUCTOR == PortfolioConstructor.CONFIDENCE_WEIGHTED:
            self.set_portfolio_construction(
                ConfidenceWeightedPortfolioConstructionModel(
                self.date_rules.every_day(), PortfolioBias.LONG_SHORT
                ))

        elif PORTFOLIO_CONSTRUCTOR == PortfolioConstructor.EQUAL_WEIGHTING:
            self.set_portfolio_construction(
                EqualWeightingPortfolioConstructionModel(
                self.date_rules.every_day(), PortfolioBias.LONG_SHORT
                ))

        elif PORTFOLIO_CONSTRUCTOR == PortfolioConstructor.INSIGHT_WEIGHTING:
            self.set_portfolio_construction(
                InsightWeightingPortfolioConstructionModel(
                self.date_rules.every_day(), PortfolioBias.LONG_SHORT
                ))
        else:
            self.error('Invalid Portfolio Constructor')
            self.quit()

        # Orders Execution
        if ORDER_EXECUTION == OrderExecutionType.IMMEDIATE:
            self.set_execution(ImmediateExecutionModel())

        elif ORDER_EXECUTION == OrderExecutionType.VOLUME_WEIGHTED_AVERAGE_PRICE:
            self.set_execution(VolumeWeightedAveragePriceExecutionModel())

        else:
            self.error('Invalid Order Execution Mode')
            self.quit()

        
    def OnData(self, data):
        pass

    def OnOrderEvent(self, orderEvent):
        order = self.transactions.get_order_by_id(orderEvent.order_id)
        if orderEvent.status == OrderStatus.FILLED:
            #self.debug(f'{self.time} | {order.symbol} | {order.direction} | {order.type} | {order.price:.2f} | {order.absolute_quantity} | ')
            self.debug(f'{orderEvent}')

    # Initializing Universe Securities
    def OnSecuritiesChanged(self, changes):
        for s in changes.AddedSecurities:

            if s.Symbol not in self.symbols_universe:

                self.log("New Symbol: %s" % s.Symbol)

                self.symbols_universe[s.Symbol] = AssetData(self, s, RESOLUTION,
                        STOPS_TARGET, STOPS_TARGET_STRONG, STRONG_SIGNAL_WEIGHT_MULTIPLIER,
                        FEATURES_SIZE, STOPS_CHECK_TARGET, 
                        MINIMUM_TRAIN_SIZE, MAXIMUM_TRAIN_SIZE, 
                        CONFIDENCE_CUTOFF, INDICATOR_CATEGORY_B, INDICATOR_CATEGORY_C,
                        TRAINING_FREQUENCY, SIGNAL_FREQUENCY, 
                        GetModelMLPClassifier(HIDDEN_LAYERS),
                        SIGNAL_SAME_DIRECTION, SIGNAL_OPPOSITE_DIRECTION, SIGNAL_NEUTRAL,
                        MAX_WEIGHT)

        for s in changes.RemovedSecurities:
            
            self.symbols_universe[s.Symbol].Stop()
            self.symbols_universe.pop(s.Symbol)
            self.log(f'Symbol Removed from Universe: {s.Symbol}')
