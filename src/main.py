import logging.config
import os
import json
from math import sqrt, exp, pi
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

class ConfigurationFactory:

    @staticmethod
    def create_config():
        conf = os.path.join("../conf", "conf.json")
        if os.path.exists(conf) and os.path.isfile(conf):
            with open(conf, "r") as f:
                config = json.load(f)
        else:
            log.info("Please check run configurations with python interpreter {PROJECT_LOC}/{PROJECT}")
        return config

    @staticmethod
    def _configure_log():
        logconfjson = os.path.join("../conf", "conf.json")
        if os.path.exists(logconfjson) and os.path.isfile(logconfjson):
            with open(logconfjson, "r") as f:
                config = json.load(f)
            logging.config.dictConfig(config["log"])
        else:
            logging.basicConfig(level=logging.INFO)

class TraderUtility:

    @staticmethod
    def compute_zeta(alpha, b, phi, k):
        return (alpha - (b/2) + np.sqrt(phi * k)) / (alpha - (b/2) - np.sqrt(phi * k))

    @staticmethod
    def compute_inventory(no_of_shares, alpha, b, phi, k, time_scale=1):
        #Create container for results
        res = []
        R = no_of_shares
        time_steps = np.linspace(0, time_scale, 50)
        zeta = TraderUtility.compute_zeta(alpha, b, phi, k)
        gamma = np.sqrt(phi / k)
        for t in time_steps:
            T_t = time_scale - t
            sinh_gamma_T_t = (zeta * exp(gamma * (T_t))) - exp(-gamma * (T_t))
            sinh_gamma_T =  (zeta * exp(gamma * (time_scale))) - exp(-gamma * (time_scale))
            inventory = R * (sinh_gamma_T_t / sinh_gamma_T)
            #inventory = 0 if inventory == 0 else inventory
            res.append(inventory)
        return np.array(res), time_steps

    @staticmethod
    def compute_trading_speed(no_of_shares, alpha, b, phi, k, time_scale=1):
        #Create container for results
        res = []
        R = no_of_shares
        time_steps = np.linspace(0, time_scale, 50)
        zeta = TraderUtility.compute_zeta(alpha, b, phi, k)
        gamma = np.sqrt(phi / k)
        for t in time_steps:
            T_t = time_scale - t
            coshh_gamma_T_t = gamma * ((zeta * exp(gamma * (T_t))) + exp(-gamma * (T_t)))
            sinh_gamma_T =  (zeta * exp(gamma * (time_scale))) - exp(-gamma * (time_scale))
            trading_speed = R * (coshh_gamma_T_t / sinh_gamma_T)
            #inventory = 0 if inventory == 0 else inventory
            res.append(trading_speed)
        return np.array(res), time_steps

class OptimalTradingStrategy:

    def __init__(self):
        self._logger = logging.getLogger("sc_logger")
        self._init_params()
        self._gen_inventory()
        self._gen_trading_speed()

    def _init_params(self):
        self._params = ConfigurationFactory.create_config()["model"]
        self._logger.info("Initialised config params: {}".format(self._params))
        self.trading_speed = {}
        self.inventory = {}

    def _gen_inventory(self):
        for p in self._params['phi']:
            self._logger.debug("Computing inventory paramaters for phi = {}".format(p))
            axis = TraderUtility.compute_inventory(100,
                                               1000000000,
                                               self._params['b'],
                                               p,
                                               self._params['k'])
            self.inventory[p] = {"inventory" : axis[0],
                                 "time" : axis[1]}

    def _gen_trading_speed(self):
        for p in self._params['phi']:
            self._logger.debug("Computing trading speed paramaters for phi = {}".format(p))
            axis = TraderUtility.compute_trading_speed(100,
                                               1000000000,
                                               self._params['b'],
                                               p,
                                               self._params['k'])
            self.trading_speed[p] = {"trading_speed" : axis[0],
                                 "time" : axis[1]}

class PlottingEngine:

    def __init__(self):
        self.desc = "Inventory Levels vs Time."

    @staticmethod
    def plot_inventory(model):
        for p in model.inventory:
            plt.plot(model.inventory[p]["inventory"],
                     model.inventory[p]["time"],
                     label=r'$\phi = {}$'.format(p))
        # chart rendering
        plt.title("Inventory vs. Time")
        plt.minorticks_on()
        ax = plt.gca()
        ax.set_facecolor(color='whitesmoke')
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.legend()
        plt.show()

    @staticmethod
    def plot_inventory_and_trading_speed(model):
        #define gridspec
        grid = plt.GridSpec(1, 2, wspace=0.4, hspace=0.3)
        # Add graphs to gridspec dynamically
        plt.subplot(grid[0, 0])
        # Duplicate axes here
        ax1 = plt.gca()
        #define plot colours here
        colours = ['blueviolet', 'seagreen', 'red', 'deepskyblue']
        for idx, p in enumerate(model.inventory):
            ax1.plot(model.inventory[p]["time"],
                     model.inventory[p]["inventory"],
                     label=r'$\phi = {}$'.format(p),
                     alpha=0.5,
                     color=colours[idx])
        #add chart formatting
        plt.minorticks_on()
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Inventory')
        ax1.set_title("Inventory vs. Time")
        plt.legend()
        ax1.set_facecolor(color='whitesmoke')
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        #plot trading speed
        plt.subplot(grid[0, 1])
        ax2 = plt.gca()
        for idx, p in enumerate(model.trading_speed):
            ax2.plot(model.trading_speed[p]["time"],
                     model.trading_speed[p]["trading_speed"],
                     label=r'$\phi = {}$'.format(p),
                     color=colours[idx],
                     alpha=0.5)
        #add chart formatting
        plt.minorticks_on()
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Trading Speed')
        ax2.set_title("Trading Speed vs. Time")
        plt.legend()
        ax2.set_facecolor(color='whitesmoke')
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

        plt.show()

if __name__ == "__main__":
    ConfigurationFactory._configure_log()
    log = logging.getLogger("sc_logger")
    log.info("Initialising Program For Stochastic Control - Optimal Execution with Permanent Price Impact")
    try:
        # Initialise Optimal Trading Strategy
        strategy = OptimalTradingStrategy()
        PlottingEngine.plot_inventory_and_trading_speed(strategy)
    except Exception as e:
        print(e)