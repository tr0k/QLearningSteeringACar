import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLData;
import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.data.basic.BasicNeuralData;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.Train;
import org.encog.neural.networks.training.propagation.back.Backpropagation;

import java.text.SimpleDateFormat;
import java.util.*;


public class QDriverNew extends Controller {

	/**
	 * Gear changing constants
	 */
	private final int[] gearUp = {7500, 7500, 7500, 7500, 7500, 0};
	private final int[] gearDown = {0, 2500, 3000, 3000, 3500, 3500};

	/**
	 * Current time format
	 */
	SimpleDateFormat sdf = new SimpleDateFormat("HH:mm:ss");

	/**
	 * Max speed for a car
	 */
	double maxSpeed = 60.0;

	/**
	 * Best reward so far
	 */
	double bestReward = 0;
	double lastAchivedDistance = 0;

	/**
	 * Current state with last action for training
	 */
	double lastState[] = null;

	/**
	 * Random generator
	 */
	private Random random = new Random();

	/**
	 * Possible moves
	 */
	private final double[] possibleActions = {-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1};

	private int selectedAction = 4;
	private double qOutput;

	/**
	 * Parameters for exploration vs exploitation dilemma
	 */
	private double t, a = 1, b = 0.99, c = 0.002;

	private BasicNetwork network;

	/**
	 * Number of actions for each game made by nn
	 */
	private int step = 0;

	/**Number of restarts of the game after each crash**/
	private int gamesPlayed = 0;

	/**
	 * Learning rate
	 */
	private double learningRate = 0.1;
	private boolean explore = true;

	public QDriverNew() {
		network = new BasicNetwork();
		network.addLayer(new BasicLayer(null, true, 8));
		network.addLayer(new BasicLayer(new ActivationTANH(), true, 22));
		network.addLayer(new BasicLayer(new ActivationTANH(), true, 1));
		network.getStructure().finalizeStructure();
		network.reset();

		// Temperature value
		t = a * Math.pow(b, gamesPlayed);
	}

	/**
	 * Calculate new gear value depending on current gear and rotations
	 * per minute of car engine
	 * @param gear
	 * @param rpm
	 * @return New gear
	 */
	private int getGear(int gear, double rpm){

		// if gear is 0 (Neutral) or -1 (Rewind) just return 1
		if (gear < 1)
			return 1;

		// check if the RPM value of car is greater than the one suggested
		// to shift up the gear from the current one
		if (gear < 6 && rpm >= gearUp[gear - 1]) {
			return gear + 1;
		} else {
			// check if the RPM value of car is lower than the one suggested
			// to shift down the gear from the current one
			if (gear > 1 && rpm <= gearDown[gear-1])
				return gear - 1;
		}

		// Otherwise keep current gear
		return gear;
	}

	@Override
	public Action control(SensorModel sensorModel) {
		if (step > 0)
			observeResultingState(sensorModel);

		boolean exploreInMove = explore;

		step++;
		Action action = new Action();

		//if hit sth restart race
		if(sensorModel.getDamage() > 0)
			action.restartRace = true;

		action.gear = getGear(sensorModel.getGear(), sensorModel.getRPM());

		//get state
		double[] sensors = sensorModel.getTrackEdgeSensors();
		//last state for nn
		lastState = getCurrentState(sensors, possibleActions[selectedAction]);

		double divisor = 0.0;
		Map<Double, Double> qValues = new HashMap<Double, Double>();
		Double bestQValue = null;

		for(int i=0; i < possibleActions.length; ++i){
			MLData input = new BasicNeuralData(getCurrentState(sensors, possibleActions[i]));
			//compute qValue for next action in current state
			MLData output = network.compute(input);

			double qValue = output.getData(0);
			qValues.put(possibleActions[i], qValue);

			if(!exploreInMove){
				//select best qValue for each action
				if (bestQValue == null || qValue > bestQValue) {
					bestQValue = qValue;
					selectedAction = i;
				}
			}
			else {
				divisor += Math.exp(qValue / t);
			}

		}

		if (exploreInMove) {
			double choice = random.nextDouble();
			double sum = 0;

			for(int i=0; i < possibleActions.length; ++i) {
				double p = Math.exp(qValues.get(possibleActions[i]) / t) / divisor;

				sum += p;
				if (choice <= sum) {
					selectedAction = i;
					break;
				}
			}
		}

		qOutput = qValues.get(possibleActions[selectedAction]);
		action.steering = possibleActions[selectedAction];

		//speed control
		if (sensorModel.getSpeed() < maxSpeed)
			action.accelerate = 1;

		return action;
	}

	private void observeResultingState(SensorModel sensorModel) {
		//Setting up reward
		double reward = 0;

		//print best reward so far
		if(sensorModel.getDistanceRaced() > bestReward) {
			bestReward = sensorModel.getDistanceRaced();
		}

		if(sensorModel.getDistanceRaced() - lastAchivedDistance > 0){
			reward += 1;
		}
		else{
			reward -= 1;
		}

		lastAchivedDistance = sensorModel.getDistanceRaced();

		if (sensorModel.getDamage() > 0)
			reward = -5;

		//observe resulting state
		double[] sensors = sensorModel.getTrackEdgeSensors();

		//getting states for every action
		List<MLData> newStates = new ArrayList<MLData>();
		for (double possibleAction : possibleActions) {
			newStates.add(new BasicNeuralData(getCurrentState(sensors, possibleAction)));
		}

		// observe next state and choosing the best qValue
		Double best = 0.0;
		for(MLData newState : newStates) {
			final MLData output = network.compute(newState);

			if (output.getData(0) > best) {
				best = output.getData(0);
			}
		}


		// adjust the neural network
		double qTarget = (1 - learningRate) * qOutput + learningRate * (reward + best);
		System.out.println("reward: " + reward);
		System.out.println("qTarget: " + qTarget);

		NeuralDataSet trainingSet = new BasicNeuralDataSet(
				new double[][]{ lastState },
				new double[][] { {qTarget} }
		);
		final Train train = new Backpropagation(network, trainingSet, 0.2, 0.9);
		train.iteration();
	}

	/**
	 * Get current state of the game
	 * @param sensors Readings from car's sensors
	 * @param action Action
     * @return Current state
     */
	private double[] getCurrentState(double[] sensors, double action){
		double frontSensor = Math.max(Math.max(sensors[8], sensors[10]), sensors[9]);
		return new double[] {
				sensors[0] / 200,
				sensors[3] / 200,
				sensors[6] / 200,
				frontSensor / 200,
				sensors[12] / 200,
				sensors[15] / 200,
				sensors[18] / 200,
				action
		};
	}

	@Override
	public void reset() {
		step = 1;
		++gamesPlayed;

		System.out.println("Games played: " + gamesPlayed);
		Calendar cal = Calendar.getInstance();
		System.out.println("["+ sdf.format(cal.getTime())+ "] Last dist: " + lastAchivedDistance);
		System.out.println("["+ sdf.format(cal.getTime())+ "] Best reward: " + bestReward);

		t = a * Math.pow(b, gamesPlayed);
		explore = (t > c);
	}

	@Override
	public void shutdown() {
		System.out.println("Driver says: Race abandoned");
	}
}