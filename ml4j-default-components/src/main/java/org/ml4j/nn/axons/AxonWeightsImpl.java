package org.ml4j.nn.axons;

import org.ml4j.EditableMatrix;
import org.ml4j.Matrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class AxonWeightsImpl implements AxonWeights {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private static final Logger LOGGER = LoggerFactory.getLogger(AxonWeightsImpl.class);

	private int inputNeuronCount;
	private int outputNeuronCount;
	private EditableMatrix leftToRightBiases;
	private EditableMatrix rightToLeftBiases;
	private EditableMatrix connectionWeights;

	public AxonWeightsImpl(int inputNeuronCount, int outputNeuronCount, Matrix connectionWeights,
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
	public Matrix applyToRightToLeftInput(Matrix input) {

		EditableMatrix output = connectionWeights.transpose().mmul(input).asEditableMatrix();
		if (rightToLeftBiases != null) {
			output.addiColumnVector(rightToLeftBiases);
		}
		return output;
	}

	@Override
	public Matrix applyToLeftToRightInput(Matrix input) {
		EditableMatrix output = connectionWeights.mmul(input).asEditableMatrix();
		if (leftToRightBiases != null) {
			output.addiColumnVector(leftToRightBiases);
		}
		return output;
	}

	@Override
	public AxonWeights dup() {
		return new AxonWeightsImpl(inputNeuronCount, outputNeuronCount, connectionWeights.dup(),
				leftToRightBiases == null ? null : leftToRightBiases.dup(),
				rightToLeftBiases == null ? null : rightToLeftBiases.dup());
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
