/*
 * Copyright 2019 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */
package org.ml4j.nn.neurons;

import java.util.Arrays;
import java.util.Objects;

import org.ml4j.FloatModifier;
import org.ml4j.FloatPredicate;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.ml4j.nn.neurons.format.features.Dimension;
import org.ml4j.nn.neurons.format.features.DimensionScope;
import org.ml4j.nn.neurons.format.features.FlatFeaturesFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Mock implementation of NeuronsActivation
 * 
 * @author Michael Lavelle
 */
public class DummyNeuronsActivation implements NeuronsActivation {

	private static final Logger LOGGER = LoggerFactory.getLogger(DummyNeuronsActivation.class);

	protected Neurons neurons;
	protected int examples;
	protected NeuronsActivationFeatureOrientation featureOrientation;

	/**
	 * @param neurons            The neurons that have been activated.
	 * @param featureOrientation The feature orientation of the matrix
	 *                           representation of these activations.
	 * @param examples           The number of training examples that caused the
	 *                           activations.
	 */
	public DummyNeuronsActivation(Neurons neurons, NeuronsActivationFeatureOrientation featureOrientation,
			int examples) {
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
	public ImageNeuronsActivation asImageNeuronsActivation(Neurons3D neurons, DimensionScope dimensionScope) {
		return new DummyImageNeuronsActivation(neurons, featureOrientation, examples);
	}

	@Override
	public void close() {
		// No-op
	}

	@Override
	public void combineFeaturesInline(NeuronsActivation arg0, MatrixFactory matrixFactory) {
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

	@Override
	public void reshape(int featureCount, int exampleCount) {
		this.neurons = new Neurons(featureCount, neurons.hasBiasUnit());
		this.examples = exampleCount;
	}
	
	@Override
	public NeuronsActivationFormat<?> getFormat() {
		return new NeuronsActivationFormat<>(featureOrientation, new FlatFeaturesFormat(), Arrays.asList(Dimension.EXAMPLE));
	}
}
