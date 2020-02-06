package org.ml4j.nn.axons;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class AxonWeightsBase implements AxonWeights {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private static final Logger LOGGER = LoggerFactory.getLogger(AxonWeightsBase.class);

	protected int inputNeuronCount;
	protected int outputNeuronCount;
	protected BiasMatrix leftToRightBiases;
	protected BiasMatrix rightToLeftBiases;
	protected WeightsMatrix connectionWeights;
	protected AxonWeightsType type;

	public AxonWeightsBase(int inputNeuronCount, int outputNeuronCount, WeightsMatrix connectionWeights,
			BiasMatrix leftToRightBiases, BiasMatrix rightToLeftBiases, AxonWeightsType type) {
		super();
		this.inputNeuronCount = inputNeuronCount;
		this.outputNeuronCount = outputNeuronCount;
		this.leftToRightBiases = leftToRightBiases;
		this.rightToLeftBiases = rightToLeftBiases;
		this.connectionWeights = connectionWeights;
		this.type = type;
	}

	public AxonWeightsType getType() {
		return type;
	}

	@Override
	public void adjustWeights(AxonWeightsAdjustment axonWeightsAdjustment,
			AxonWeightsAdjustmentDirection adjustmentDirection) {
		if (adjustmentDirection == AxonWeightsAdjustmentDirection.ADDITION) {
			LOGGER.debug("Adding adjustment to axon weights");
			connectionWeights.getWeights().asEditableMatrix().addi(axonWeightsAdjustment.getConnectionWeights());
			if (axonWeightsAdjustment.getLeftToRightBiases().isPresent()) {
				leftToRightBiases.getWeights().asEditableMatrix().addi(axonWeightsAdjustment.getLeftToRightBiases().get());
			}
			if (axonWeightsAdjustment.getRightToLeftBiases().isPresent()) {
				rightToLeftBiases.getWeights().asEditableMatrix().addi(axonWeightsAdjustment.getRightToLeftBiases().get());
			}
		} else {
			LOGGER.debug("Subtracting adjustment from axon weights");
			connectionWeights.getWeights().asEditableMatrix().subi(axonWeightsAdjustment.getConnectionWeights());
			if (axonWeightsAdjustment.getLeftToRightBiases().isPresent()) {
				leftToRightBiases.getWeights().asEditableMatrix().subi(axonWeightsAdjustment.getLeftToRightBiases().get());
			}
			if (axonWeightsAdjustment.getRightToLeftBiases().isPresent()) {
				rightToLeftBiases.getWeights().asEditableMatrix().subi(axonWeightsAdjustment.getRightToLeftBiases().get());
			}
		}
	}

	@Override
	public WeightsMatrix getConnectionWeights() {
		return connectionWeights;
	}

	@Override
	public int getInputNeuronCount() {
		return inputNeuronCount;
	}

	@Override
	public BiasMatrix getLeftToRightBiases() {
		return leftToRightBiases;
	}

	@Override
	public int getOutputNeuronsCount() {
		return outputNeuronCount;
	}

	@Override
	public BiasMatrix getRightToLeftBiases() {
		return rightToLeftBiases;
	}

}
