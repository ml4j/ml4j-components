package org.ml4j.nn.axons;

import java.util.Optional;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.Neurons;

public class DefaultScaleAndShiftAxonWeightsInitialiser implements AxonWeightsInitialiser {

	private Neurons neurons;

	public DefaultScaleAndShiftAxonWeightsInitialiser(Neurons neurons) {
		this.neurons = neurons;
	}

	@Override
	public Matrix getInitialConnectionWeights(MatrixFactory matrixFactory) {
		return matrixFactory.createOnes(neurons.getNeuronCountExcludingBias(), 1);

	}

	@Override
	public Optional<Matrix> getInitialLeftToRightBiases(MatrixFactory matrixFactory) {
		return Optional
				.of(matrixFactory.createOnes(neurons.getNeuronCountExcludingBias(), 1).asEditableMatrix().muli(0.01f));

	}

	@Override
	public Optional<Matrix> getInitialRightToLeftBiases(MatrixFactory matrixFactory) {
		return Optional.empty();
	}
}
