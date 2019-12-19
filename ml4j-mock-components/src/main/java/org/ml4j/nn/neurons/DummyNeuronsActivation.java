package org.ml4j.nn.neurons;

import java.util.Objects;

import org.ml4j.FloatModifier;
import org.ml4j.FloatPredicate;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DummyNeuronsActivation implements NeuronsActivation {
	
	private static final Logger LOGGER = LoggerFactory.getLogger(DummyNeuronsActivation.class);
	
	protected Neurons neurons;
	protected int examples;
	protected NeuronsActivationFeatureOrientation featureOrientation;
	
	public DummyNeuronsActivation(Neurons neurons, NeuronsActivationFeatureOrientation featureOrientation, int examples) {
		this.neurons = neurons;
		this.examples = examples;
		this.featureOrientation = featureOrientation;
		Objects.requireNonNull(neurons, "neurons");
	}

	@Override
	public void addInline(MatrixFactory arg0, NeuronsActivation arg1) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void applyValueModifier(FloatModifier arg0) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void applyValueModifier(FloatPredicate arg0, FloatModifier arg1) {
		throw new UnsupportedOperationException();
	}

	@Override
	public ImageNeuronsActivation asImageNeuronsActivation(Neurons3D neurons) {
		return new DummyImageNeuronsActivation(neurons, featureOrientation, examples);
	}

	@Override
	public void close() {

	}

	@Override
	public void combineFeaturesInline(NeuronsActivation arg0) {
		throw new UnsupportedOperationException();
	}

	@Override
	public NeuronsActivation dup() {
		return new DummyNeuronsActivation(neurons, featureOrientation, examples);
	}

	@Override
	public Matrix getActivations(MatrixFactory matrixFactory) {
		LOGGER.debug("Creating activations matrix from neurons activation");
		return matrixFactory.createMatrix(neurons.getNeuronCountExcludingBias(), examples);
	}

	@Override
	public int getColumns() {
		if (featureOrientation == NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET) {
			return getFeatureCount();
		} else {
			return getExampleCount();
		}
	}

	@Override
	public int getExampleCount() {
		return examples;
	}

	@Override
	public int getFeatureCount() {
		return neurons.getNeuronCountExcludingBias();
	}

	@Override
	public NeuronsActivationFeatureOrientation getFeatureOrientation() {
		return featureOrientation;
	}

	@Override
	public Neurons getNeurons() {
		return neurons;
	}

	@Override
	public int getRows() {
		if (featureOrientation == NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
			return getFeatureCount();
		} else {
			return getExampleCount();
		}
	}

	@Override
	public boolean isImmutable() {
		throw new UnsupportedOperationException();
	}

	@Override
	public void setImmutable(boolean arg0) {
		throw new UnsupportedOperationException();
	}

}
