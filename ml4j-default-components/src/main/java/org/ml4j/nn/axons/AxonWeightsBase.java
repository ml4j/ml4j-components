package org.ml4j.nn.axons;

import org.ml4j.EditableMatrix;
import org.ml4j.Matrix;
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
	protected EditableMatrix leftToRightBiases;
	protected EditableMatrix rightToLeftBiases;
	protected EditableMatrix connectionWeights;

	public AxonWeightsBase(int inputNeuronCount, int outputNeuronCount, Matrix connectionWeights,
			Matrix leftToRightBiases, Matrix rightToLeftBiases) {
		super();
		this.inputNeuronCount = inputNeuronCount;
		this.outputNeuronCount = outputNeuronCount;
		this.leftToRightBiases = leftToRightBiases == null ? null : leftToRightBiases.asEditableMatrix();
		this.rightToLeftBiases = rightToLeftBiases == null ? null : rightToLeftBiases.asEditableMatrix();
		this.connectionWeights = connectionWeights.asEditableMatrix();
	}

	@Override
	public void adjustWeights(AxonWeightsAdjustment axonWeightsAdjustment,
			AxonWeightsAdjustmentDirection adjustmentDirection) {
		if (adjustmentDirection == AxonWeightsAdjustmentDirection.ADDITION) {
			LOGGER.debug("Adding adjustment to axon weights");
			connectionWeights.addi(axonWeightsAdjustment.getConnectionWeights());
			if (axonWeightsAdjustment.getLeftToRightBiases().isPresent()) {
				leftToRightBiases.addi(axonWeightsAdjustment.getLeftToRightBiases().get());
			}
			if (axonWeightsAdjustment.getRightToLeftBiases().isPresent()) {
				rightToLeftBiases.addi(axonWeightsAdjustment.getRightToLeftBiases().get());
			}
		} else {
			LOGGER.debug("Subtracting adjustment from axon weights");
			connectionWeights.subi(axonWeightsAdjustment.getConnectionWeights());
			if (axonWeightsAdjustment.getLeftToRightBiases().isPresent()) {
				leftToRightBiases.subi(axonWeightsAdjustment.getLeftToRightBiases().get());
			}
			if (axonWeightsAdjustment.getRightToLeftBiases().isPresent()) {
				rightToLeftBiases.subi(axonWeightsAdjustment.getRightToLeftBiases().get());
			}
		}
	}

	@Override
	public Matrix getConnectionWeights() {
		return connectionWeights;
	}

	@Override
	public int getInputNeuronCount() {
		return inputNeuronCount;
	}

	@Override
	public Matrix getLeftToRightBiases() {
		return leftToRightBiases;
	}

	@Override
	public int getOutputNeuronsCount() {
		return outputNeuronCount;
	}

	@Override
	public Matrix getRightToLeftBiases() {
		return rightToLeftBiases;
	}

}
