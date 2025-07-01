# region imports
from AlgorithmImports import *
from enumerators import (SignalOnSameDirection, SignalOnOppositeDirection, SignalNeutral, 
                        SIGNAL_LONG, SIGNAL_SHORT,
                        SIGNAL_NONE, SIGNAL_LONG_STRONG, SIGNAL_SHORT_STRONG)
# endregion

def SignalHandler(algorithm, symbol, signal, confidence, weight, 
                  signal_same_direction, signal_opp_direction, signal_neutral,
                  resolution, bars):

    insight_period = timedelta(seconds=1)
    if resolution == Resolution.MINUTE:
        insight_period = timedelta(minutes=bars)
    elif resolution == Resolution.HOUR:
        insight_period = timedelta(hours=bars)
    elif resolution == Resolution.DAILY:
        insight_period = timedelta(days=bars)

    if signal == SIGNAL_NONE:

        # None Signal and Some Position
        if algorithm.portfolio[symbol].invested:

            if signal_neutral == SignalNeutral.CLOSE_IF_PROFITABLE:
                if algorithm.portfolio[symbol].profit > 0:
                    algorithm.debug(f'Closing Profitable Position: {symbol}')
                    algorithm.liquidate(symbol)
                    return

            if signal_neutral == SignalNeutral.CLOSE:
                algorithm.debug(f'Flat Signal Closing Position: {symbol}')
                algorithm.liquidate(symbol)
                return

            if signal_neutral == SignalNeutral.EMMIT_SIGNAL:
                algorithm.emit_insights(Insight.price(symbol,
                        period=insight_period,
                        direction=InsightDirection.FLAT, 
                        confidence=confidence,
                        weight=weight
                        ))
                return

    elif signal == SIGNAL_LONG:

        # Long Signal and No Position
        if not algorithm.portfolio[symbol].invested:
            algorithm.debug(f'Enter Long: {symbol}')
            algorithm.emit_insights(Insight.price(symbol,
                period=insight_period,
                direction=InsightDirection.UP,
                confidence=confidence,
                weight=weight
                ))
            return

        # Long Signal and Long Position
        elif algorithm.portfolio[symbol].is_long and signal_same_direction == SignalOnSameDirection.EMMIT_SIGNAL:
            algorithm.debug(f'Increase Long: {symbol}')
            algorithm.emit_insights(Insight.price(symbol,
                period=insight_period,
                direction=InsightDirection.UP, 
                confidence=confidence,
                weight=weight
                ))
            return

        # Long Signal and Long Position
        elif algorithm.portfolio[symbol].is_long and signal_same_direction == SignalOnSameDirection.EMMIT_IF_NOT_PROFITABLE:
            if algorithm.portfolio[symbol].profit < 0:
                algorithm.debug(f'Increase Non-Profitable Long: {symbol}')
                algorithm.emit_insights(Insight.price(symbol,
                    period=insight_period,
                    direction=InsightDirection.UP, 
                    confidence=confidence,
                    weight=weight
                    ))
                return

        # Long Signal and Short Position
        elif algorithm.portfolio[symbol].is_short and signal_opp_direction == SignalOnOppositeDirection.CLOSE:
            algorithm.debug(f'Closing Short: {symbol}')
            algorithm.liquidate(symbol)
            return

        # Long Signal and Short Position
        elif algorithm.portfolio[symbol].is_short and signal_opp_direction == SignalOnOppositeDirection.EMMIT_SIGNAL:
            algorithm.debug(f'Emmit Long Signal for Short Position: {symbol}')
            algorithm.emit_insights(Insight.price(symbol,
                    period=insight_period,
                    direction=InsightDirection.DOWN, 
                    confidence=confidence,
                    weight=weight
                    ))
            return

        # Long Signal and Short Position
        elif algorithm.portfolio[symbol].is_short and signal_opp_direction == SignalOnOppositeDirection.CLOSE_IF_PROFITABLE:
            if algorithm.portfolio[symbol].profit > 0:
                algorithm.debug(f'Closing Profitable Short: {symbol}')
                algorithm.liquidate(symbol)
            return

    elif signal == SIGNAL_SHORT:

        # Short Signal and No Position
        if not algorithm.portfolio[symbol].invested:
            algorithm.debug(f'Enter Short: {symbol}')
            algorithm.emit_insights(Insight.price(symbol,
                    period=insight_period,
                    direction=InsightDirection.DOWN, 
                    confidence=confidence,
                    weight=weight
                    ))
            return

        # Short Signal and Short Position
        elif algorithm.portfolio[symbol].is_short and signal_same_direction == SignalOnSameDirection.EMMIT_SIGNAL:
            algorithm.debug(f'Increase Short: {symbol}')
            algorithm.emit_insights(Insight.price(symbol,
                    period=insight_period,
                    direction=InsightDirection.DOWN, 
                    confidence=confidence,
                    weight=weight
                    ))
            return

        # Short Signal and Short Position
        elif algorithm.portfolio[symbol].is_short and signal_same_direction == SignalOnSameDirection.EMMIT_IF_NOT_PROFITABLE:
            if algorithm.portfolio[symbol].profit < 0:
                algorithm.debug(f'Increase Non-Profitable Short: {symbol}')
                algorithm.emit_insights(Insight.price(symbol,
                        period=insight_period,
                        direction=InsightDirection.DOWN, 
                        confidence=confidence,
                        weight=weight
                        ))
                return

        # Short Signal and Long Position
        elif algorithm.portfolio[symbol].is_long and signal_opp_direction == SignalOnOppositeDirection.CLOSE:
            algorithm.debug(f'Closing Long: {symbol}')
            algorithm.liquidate(symbol)
            return

        # Short Signal and Long Position
        elif algorithm.portfolio[symbol].is_long and signal_opp_direction == SignalOnOppositeDirection.EMMIT_SIGNAL:
            algorithm.debug(f'Emmit Short Signal for Long Position: {symbol}')
            algorithm.emit_insights(Insight.price(symbol,
                period=insight_period,
                direction=InsightDirection.DOWN, 
                confidence=confidence,
                weight=weight
                ))
            return

        # Short Signal and Long Position
        elif algorithm.portfolio[symbol].is_long and signal_opp_direction == SignalOnOppositeDirection.CLOSE_IF_PROFITABLE:
            if algorithm.portfolio[symbol].profit  > 0:
                algorithm.debug(f'Closing Profitable Long: {symbol}')
                algorithm.liquidate(symbol)
            return

    elif signal == SIGNAL_LONG_STRONG:
        # Strong Long Signal and No Position
        if not algorithm.portfolio[symbol].invested:
            algorithm.debug(f'Enter Long (Strong Signal): {symbol}')
            algorithm.emit_insights(Insight.price(symbol,
                period=insight_period,
                direction=InsightDirection.UP, 
                confidence=confidence,
                weight=weight
                ))
            return
        
        # Strong Long Signal and Long Position
        elif algorithm.portfolio[symbol].is_long and signal_same_direction == SignalOnSameDirection.EMMIT_IF_STRONG_SIGNAL:
            algorithm.debug(f'Increase Long (Strong Signal): {symbol}')
            algorithm.emit_insights(Insight.price(symbol,
                period=insight_period,
                direction=InsightDirection.UP, 
                confidence=confidence,
                weight=weight
                ))
            return

        # Strong Long Signal and Long Position (profit < 0)
        elif algorithm.portfolio[symbol].is_long and signal_same_direction == SignalOnSameDirection.EMMIT_IF_STRONG_SIGNAL_AND_NOT_PROFITABLE:

            if algorithm.portfolio[symbol].profit < 0:
                algorithm.debug(f'Increase Long (Strong Signal) for non-profitable position: {symbol}')
                algorithm.emit_insights(Insight.price(symbol,
                    period=insight_period,
                    direction=InsightDirection.UP, 
                    confidence=confidence,
                    weight=weight
                    ))
                return

        # Strong Long Signal and Short Position
        elif algorithm.portfolio[symbol].is_short and signal_opp_direction == SignalOnOppositeDirection.EMMIT_IF_STRONG_SIGNAL:
            algorithm.debug(f'Emmit Strong Long Signal for Short Position: {symbol}')
            algorithm.emit_insights(Insight.price(symbol,
                    period=insight_period,
                    direction=InsightDirection.UP, 
                    confidence=confidence,
                    weight=weight
                    ))
            return

    elif signal == SIGNAL_SHORT_STRONG:
        # Strong Short Signal and No Position
        if not algorithm.portfolio[symbol].invested:
            algorithm.debug(f'Enter Short (Strong Signal): {symbol}')
            algorithm.emit_insights(Insight.price(symbol,
                    period=insight_period,
                    direction=InsightDirection.DOWN, 
                    confidence=confidence,
                    weight=weight
                    ))
            return

        # Strong Short Signal and Short Position
        elif algorithm.portfolio[symbol].is_short and signal_same_direction == SignalOnSameDirection.EMMIT_IF_STRONG_SIGNAL:
            algorithm.debug(f'Increase Short (Strong Signal): {symbol}')
            algorithm.emit_insights(Insight.price(symbol,
                    period=insight_period,
                    direction=InsightDirection.DOWN, 
                    confidence=confidence,
                    weight=weight
                    ))
            return

        # Strong Short Signal and Short Position (profit < 0)
        elif algorithm.portfolio[symbol].is_short and signal_same_direction == SignalOnSameDirection.EMMIT_IF_STRONG_SIGNAL_AND_NOT_PROFITABLE:

            if algorithm.portfolio[symbol].profit < 0:
                algorithm.debug(f'Increase Short (Strong Signal) for non-profitable position: {symbol}')
                algorithm.emit_insights(Insight.price(symbol,
                        period=insight_period,
                        direction=InsightDirection.DOWN, 
                        confidence=confidence,
                        weight=weight
                        ))
                return

        # Strong Short Signal and Long Position
        elif algorithm.portfolio[symbol].is_long and signal_opp_direction == SignalOnOppositeDirection.EMMIT_IF_STRONG_SIGNAL:
            algorithm.debug(f'Emmit Strong Short Signal for Long Position: {symbol}')
            algorithm.emit_insights(Insight.price(symbol,
                period=insight_period,
                direction=InsightDirection.DOWN, 
                confidence=confidence,
                weight=weight
                ))
            return
