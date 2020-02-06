package org.ml4j.nn.axons;

import org.ml4j.EditableMatrix;
import org.ml4j.Matrix;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ScaleAndShiftAxonWeightsImpl extends AxonWeightsBase implements AxonWeights {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	@SuppressWarnings("unused")
	private static final Logger LOGGER = LoggerFactory.getLogger(ScaleAndShiftAxonWeightsImpl.class);

	public ScaleAndShiftAxonWeightsImpl(int inputNeuronCount, int outputNeuronCount, WeightsMatrix connectionWeights,
			BiasMatrix leftToRightBiases, BiasMatrix rightToLeftBiases) {
		super(inputNeuronCount, outputNeuronCount, connectionWeights, leftToRightBiases, rightToLeftBiases,
				AxonWeightsType.SCALE_AND_SHIFT);
	}

	@Override
	public NeuronsActivation applyToRightToLeftInput(NeuronsActivation input, AxonsContext axonsContext) {

		Matrix axonWeightsInput = input.getActivations(axonsContext.getMatrixFactory());
		
		EditableMatrix output = axonWeightsInput.mulColumnVector(connectionWeights.getWeights()).asEditableMatrix();
		if (rightToLeftBiases != null) {
			output.addiColumnVector(rightToLeftBiases.getWeights());
		}
		return new NeuronsActivationImpl(new Neurons(this.inputNeuronCount, false), output, input.getFormat());
	}

	@Override
	public NeuronsActivation applyToLeftToRightInput(NeuronsActivation input, AxonsContext axonsContext) {
		
		Matrix axonWeightsInput = input.getActivations(axonsContext.getMatrixFactory());
		
		EditableMatrix output = axonWeightsInput.mulColumnVector(connectionWeights.getWeights()).asEditableMatrix();
		if (leftToRightBiases != null) {
			output.addiColumnVector(leftToRightBiases.getWeights());
		}
		return new NeuronsActivationImpl(new Neurons(this.outputNeuronCount, false), output, input.getFormat());
	}

	@Override
	public AxonWeights dup() {
		return new ScaleAndShiftAxonWeightsImpl(inputNeuronCount, outputNeuronCount, connectionWeights.dup(),
				leftToRightBiases == null ? null : leftToRightBiases.dup(),
				rightToLeftBiases == null ? null : rightToLeftBiases.dup());
	}

}
