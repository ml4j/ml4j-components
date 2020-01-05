package org.ml4j.nn.axons;

import org.ml4j.EditableMatrix;
import org.ml4j.Matrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ScaleAndShiftAxonWeightsImpl extends AxonWeightsBase implements AxonWeights {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	@SuppressWarnings("unused")
	private static final Logger LOGGER = LoggerFactory.getLogger(ScaleAndShiftAxonWeightsImpl.class);

	public ScaleAndShiftAxonWeightsImpl(int inputNeuronCount, int outputNeuronCount, Matrix connectionWeights,
			Matrix leftToRightBiases, Matrix rightToLeftBiases) {
		super(inputNeuronCount, outputNeuronCount, connectionWeights, leftToRightBiases, rightToLeftBiases);
	}

	
	@Override
	public Matrix applyToRightToLeftInput(Matrix input) {

		EditableMatrix output = input.mulColumnVector(connectionWeights).asEditableMatrix();
		if (rightToLeftBiases != null) {
			output.addiColumnVector(rightToLeftBiases);
		}
		return output;
	}

	@Override
	public Matrix applyToLeftToRightInput(Matrix input) {
		EditableMatrix output = input.mulColumnVector(connectionWeights).asEditableMatrix();
		if (leftToRightBiases != null) {
			output.addiColumnVector(leftToRightBiases);
		}
		return output;
	}

	@Override
	public AxonWeights dup() {
		return new ScaleAndShiftAxonWeightsImpl(inputNeuronCount, outputNeuronCount, connectionWeights.dup(),
				leftToRightBiases == null ? null : leftToRightBiases.dup(),
				rightToLeftBiases == null ? null : rightToLeftBiases.dup());
	}


}
