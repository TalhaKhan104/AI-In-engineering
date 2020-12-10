import pandas as pd



#-------------------------------------------------------------------------------------------- Symeon ----------------------------------------------------#
df = pd.read_parquet("../Operation_OCV/stock_data/stock_data_eod/bank/BAC")
df = df[["adjusted_volume", "adjusted_close", "adjusted_open", "adjusted_high", "adjusted_low"]]
df.columns = ["volume", "close", "open", "high", "low"]
