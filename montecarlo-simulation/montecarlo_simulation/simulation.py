import argparse

from montecarlo_simulation import draw_pbox, calculate_inundation, CDF_emprical_distribution, \
    montecarlo_inundation_simulation, DataLoader

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("lowerbound")
    parser.add_argument("upperbound")

    args = parser.parse_args()

    data_loader = DataLoader(args.filename)

    simulations = montecarlo_inundation_simulation(data_loader.data['Base_Fresh'], data_loader.data['ff_height'],
                                                   num_simulations=10, num_levels=2)
    inundation = calculate_inundation(data_loader.data["Base_Fresh"], data_loader.data["ff_height"])
    CDF_emprical_distribution(inundation, show=False)

    for simulation in simulations:
        CDF_emprical_distribution(simulation["difference"], show=False, label=None)

    upper_band_data_looader = DataLoader(args.upperbound, columns=["x"])
    lower_band_data_looader = DataLoader(args.lowerbound, columns=["x"])

    xlabel = 'Inundation Depth (m)'
    ylabel = 'Cumulative Probability'

    draw_pbox(upper_band_data_looader.data, lower_band_data_looader.data, xlabel=xlabel, ylabel=ylabel)
