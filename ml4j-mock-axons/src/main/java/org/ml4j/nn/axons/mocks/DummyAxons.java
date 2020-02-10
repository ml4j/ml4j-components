package org.ml4j.nn.axons.mocks;

import java.util.Optional;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.AxonsType;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;

public class DummyAxons<L extends Neurons, R extends Neurons, A extends Axons<L, R, A>> implements Axons<L, R, A> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private MatrixFactory matrixFactory;
	private L leftNeurons;
	private R rightNeurons;
	private AxonsType axonsType;

	public DummyAxons(AxonsType axonsType, MatrixFactory matrixFactory, L leftNeurons, R rightNeurons) {
		this.matrixFactory = matrixFactory;
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
		this.axonsType = axonsType;
	}

	@Override
	public L getLeftNeurons() {
		return leftNeurons;
	}

	@Override
	public R getRightNeurons() {
		return rightNeurons;
	}

	@Override
	public AxonsActivation pushLeftToRight(NeuronsActivation leftNeuronsActivation,
			AxonsActivation previousRightToLeftActivation, AxonsContext axonsContext) {
		Matrix output = matrixFactory.createMatrix(rightNeurons.getNeuronCountExcludingBias(),
				leftNeuronsActivation.getExampleCount());
		NeuronsActivation outputActivation = new NeuronsActivationImpl(getRightNeurons(), output,
				leftNeuronsActivation.getFormat());
		return new DummyAxonsActivation(this, () -> leftNeuronsActivation, outputActivation);
	}

	@Override
	public AxonsActivation pushRightToLeft(NeuronsActivation rightNeuronsActivation,
			AxonsActivation previousLeftToRightActivation, AxonsContext axonsContext) {
		throw new UnsupportedOperationException();
	}

	@Override
	public A dup() {
		throw new UnsupportedOperationException();
	}

	@Override
	public boolean isTrainable(AxonsContext axonsContext) {
		return true;
	}

	@Override
	public Optional<NeuronsActivationFormat<?>> optimisedFor() {
		return Optional.empty();
	}
	
	@Override
	public boolean isSupported(NeuronsActivationFormat<?> format) {
		return NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET.equals(format.getFeatureOrientation());
	}

	@Override
	public AxonsType getAxonsType() {
		return axonsType;
	}

}
