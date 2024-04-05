import pandas as pd
   
 # Load the CSV files into pandas DataFrames
def generate_df():
    passengers_df = pd.read_csv(os.path.join(app.static_folder, 'dataverse_files', 'ttav_passengers.csv')) 
    voyages_df = pd.read_csv(os.path.join(app.static_folder, 'dataverse_files', 'ttav_voyages.csv'))
    routes_df = pd.read_csv(os.path.join(app.static_folder, 'dataverse_files', 'ttav_routes.csv'))
    occupations_df = pd.read_csv(os.path.join(app.static_folder, 'dataverse_files', 'ttav_occupations.csv'))

    # Perform the joins
    passengers_voyages_df = pd.merge(passengers_df, voyages_df, on='MID')
    voyages_routes_df = pd.merge(passengers_voyages_df, routes_df, on='routeID')
    final_df = pd.merge(voyages_routes_df, occupations_df, on='occID')