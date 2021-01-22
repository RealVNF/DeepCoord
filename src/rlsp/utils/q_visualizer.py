from keras.layers import Concatenate, Dense, Flatten, Input
from keras.models import Model, model_from_yaml
from keras.optimizers import Adam
from rlsp.envs.environment_limits import EnvironmentLimits
from spinterface.spinterface import SimulatorState
import numpy as np
import argparse


class Q_Visulaizer(object):
    def __init__(self, model_file: str, weights_file: str):

        model_file_contents = None
        with open(model_file, 'r') as yaml_file:
            model_file_contents = yaml_file.read()
        self.model = model_from_yaml(model_file_contents)
        self.weights_file = weights_file

        self.model.load_weights(weights_file)

    def predict(self, x):
        self.x = x
        output = self.model.predict(x)  # Same as method used in DDPG agent
        return output


def main():
    predictions = []
    weights_file = ('results/agent_default_conf/triangle-in1-cap10-delay10/abc/det-arrival10_det-size001_duration100/'
                    '2019-10-04_12-00-22_seed7540/weights_critic.h5f')
    model_file = ('results/agent_default_conf/triangle-in1-cap10-delay10/abc/det-arrival10_det-size001_duration100/'
                  '2019-10-04_12-00-22_seed7540/model_critic.yaml')

    # These were sampled from the simulator_wrapper. example_actions[x] results in example_states[x]
    example_actions = [
        [
            4.94131371e-02, 1.32373261e+00, -9.52624157e-02, -3.90273556e-02,
            -1.49762228e-01, 1.20653641e+00, -1.00836195e-01, 9.76924539e-01,
            -2.24551022e-01, -8.05902258e-02, 5.55979490e-01, 3.75012308e-01,
            4.66273725e-02, 1.67559981e-01, 1.20014465e+00, 8.84849191e-01,
            4.73266006e-01, -1.01047463e-03, 1.95914939e-01, 6.94492340e-01,
            4.00339425e-01, -1.44841984e-01, 1.17309049e-01, 1.09129405e+00,
            6.38748348e-01, -1.83474109e-01, -7.39084259e-02
        ],
        [
            0.0442463, 1.2896769, -0.12166985, -0.02021519, -0.14719011, 1.1884401,
            -0.07958993, 0.97225976, -0.19635911, -0.0812363, 0.5351899, 0.34173545,
            0.06046494, 0.14159602, 1.1811118, 0.895503, 0.4840171, 0.02350514,
            0.18683067, 0.66949755, 0.42556795, -0.13954563, 0.11468667, 1.0992641,
            0.6262401, -0.22523677, -0.09948803
        ]
    ]
    example_states = [
        [0.8, 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 1., 0., 1., 1., 1.],
        [0.96666667, 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 1., 0., 1., 1., 1.],
    ]
    # inputs = inputs_go_here

    q_visualizer = Q_Visulaizer(model_file, weights_file)
    # Following part inspired from
    # https://github.com/Alexander-H-Liu/Policy-Gradient-and-Actor-Critic-Keras/blob/master/agent_dir/agent_actorcritic.py

    for state_action_index in range(len(example_actions)):
        action_reshaped = np.array(np.expand_dims(example_actions[state_action_index], 0))
        state_reshaped = np.array(np.expand_dims(np.expand_dims(example_states[state_action_index], 0), 0))
        prediction = q_visualizer.predict([action_reshaped, state_reshaped])
        predictions.append(prediction)
        print(prediction)  # Just to make the build pass for now


if __name__ == "__main__":
    main()
