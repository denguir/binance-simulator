

# when get_symbol_data is called:
# 1- list all the called get_symbol_data with same (symbol, resolution)
# 2- compute intersection with called date intervals and query date interval
# 3- call cached fct for called date intervals
# 4- call api fct for the rest 

# cache decorator for load_symbol_data
# that will break the call into sub calls