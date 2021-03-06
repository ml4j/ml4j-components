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
package org.ml4j.nn.components.axons;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Optional;
import java.util.Set;

import org.ml4j.EditableMatrix;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsBaseType;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.AxonsType;
import org.ml4j.nn.axons.ScaleAndShiftAxons;
import org.ml4j.nn.components.axons.base.DirectedAxonsComponentBase;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of BatchNormDirectedAxonsComponent
 * 
 * @author Michael Lavelle
 *
 * @param <L> The type of Neurons on the left/right of this
 *            BatchNormDirectedAxonsComponent.
 */
public class DefaultBatchNormDirectedAxonsComponentImpl<L extends Neurons> extends
		DirectedAxonsComponentBase<L, L, Axons<L, L, ?>> implements BatchNormDirectedAxonsComponent<L, Axons<L, L, ?>> {

	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultDirectedAxonsComponentImpl.class);
	/**
	 * Defaut serialization id;
	 */
	private static final long serialVersionUID = 1L;

	private Matrix exponentiallyWeightedAverageInputFeatureMeans;
	private Matrix exponentiallyWeightedAverageInputFeatureVariances;
	// TODO
	private float betaForExponentiallyWeightedAverages = 0.99f;
	
	public DefaultBatchNormDirectedAxonsComponentImpl(String name, Axons<L, L, ?> axons, Matrix mean, Matrix stddev) {
		super(name, axons);
		this.exponentiallyWeightedAverageInputFeatureMeans = mean;
		this.exponentiallyWeightedAverageInputFeatureVariances = stddev;
	}

	@Override
	public float getBetaForExponentiallyWeightedAverages() {
		return betaForExponentiallyWeightedAverages;
	}

	@Override
	public Matrix getExponentiallyWeightedAverageInputFeatureMeans() {
		return exponentiallyWeightedAverageInputFeatureMeans;
	}

	@Override
	public Matrix getExponentiallyWeightedAverageInputFeatureVariances() {
		return exponentiallyWeightedAverageInputFeatureVariances;
	}

	@Override
	public void setExponentiallyWeightedAverageInputFeatureMeans(Matrix exponentiallyWeightedAverageInputFeatureMeans) {
		this.exponentiallyWeightedAverageInputFeatureMeans = exponentiallyWeightedAverageInputFeatureMeans;
	}

	@Override
	public void setExponentiallyWeightedAverageInputFeatureVariances(
			Matrix exponentiallyWeightedAverageInputFeatureVariances) {
		this.exponentiallyWeightedAverageInputFeatureVariances = exponentiallyWeightedAverageInputFeatureVariances;
	}

	@Override
	public BatchNormDirectedAxonsComponent<L, Axons<L, L, ?>> dup(DirectedComponentFactory directedComponentFactory) {
		return new DefaultBatchNormDirectedAxonsComponentImpl<>(name, axons.dup(),
				exponentiallyWeightedAverageInputFeatureMeans == null ? null
						: exponentiallyWeightedAverageInputFeatureMeans.dup(),
				exponentiallyWeightedAverageInputFeatureVariances == null ? null
						: exponentiallyWeightedAverageInputFeatureVariances.dup());
	}

	@Override
	public DirectedAxonsComponentActivation forwardPropagate(NeuronsActivation input, AxonsContext axonsContext) {
		LOGGER.debug("Forward propagating through DefaultBatchNormDirectedAxonsComponentImpl:" + getName());
		
		// TODO
		/*
		 * if (neuronsActivation.getFeatureCount() !=
		 * getInputNeurons().getNeuronCountExcludingBias()) { throw new
		 * IllegalArgumentException(neuronsActivation.getFeatureCount() + ":" +
		 * getInputNeurons().getNeuronCountExcludingBias()); }
		 */
		Matrix activations = input.getActivations(axonsContext.getMatrixFactory());
		
		if (activations.isImmutable()) {
			activations = activations.dup();
		}
		Matrix meanColumnVector = getMeanColumnVector(activations, axonsContext.getMatrixFactory(),
				axonsContext.isTrainingContext());

		Matrix varianceColumnVector = getVarianceColumnVector(activations, axonsContext.getMatrixFactory(),
				meanColumnVector, axonsContext.isTrainingContext());
		Matrix xhat = activations.asEditableMatrix().subiColumnVector(meanColumnVector)
				.diviColumnVector(getStdDevColumnVector(varianceColumnVector));
		if (!input.isImmutable()) {
			input.close();
		}

		NeuronsActivation xhatN = new NeuronsActivationImpl(this.getAxons().getRightNeurons(), xhat,
				input.getFormat());

		AxonsActivation axonsActivation = axons.pushLeftToRight(xhatN, null, axonsContext);

		// TODO
		return new PrototypeBatchNormDirectedAxonsComponentActivationImpl(this, (ScaleAndShiftAxons) this.axons,
				axonsActivation, meanColumnVector, varianceColumnVector, axonsContext);
	}

	/**
	 * Naive implementation to construct a variance row vector with an entry for
	 * each feature.
	 * 
	 * @param matrix        The input matrix
	 * @param matrixFactory The matrix factory.
	 * @param meanRowVector The mean row vector.
	 * @return A row vector the the variances.
	 */
	protected Matrix getVarianceColumnVector(Matrix matrix, MatrixFactory matrixFactory, Matrix meanColumnVector,
			boolean isTraining) {

		if (isTraining) {
			if (matrix.getColumns() == 1) {
				if (exponentiallyWeightedAverageInputFeatureVariances != null) {
					return exponentiallyWeightedAverageInputFeatureVariances;
				} else {
					throw new IllegalStateException("Unable to calcuate mean and variance for batch "
							+ "norm on a single example - no exponentially weighted average available");
				}
			}
		} else {
			if (exponentiallyWeightedAverageInputFeatureVariances != null) {
				return exponentiallyWeightedAverageInputFeatureVariances;
			} else {
				throw new IllegalStateException("Unable to obtain variance for batch "
						+ "norm when not in training - rolling variance not supplied");
			}
		}

		EditableMatrix columnVector = (EditableMatrix) matrixFactory.createMatrix(matrix.getRows(), 1);
		for (int r = 0; r < matrix.getRows(); r++) {
			float total = 0f;
			float count = 0;
			for (int c = 0; c < matrix.getColumns(); c++) {
				float diff = (matrix.get(r, c) - meanColumnVector.get(r, 0));
				total = total + diff * diff;
				count++;
			}
			float variance = total / count;
			columnVector.put(r, 0, variance);
		}
		return columnVector;
	}

	protected Matrix getMeanColumnVector(Matrix matrix, MatrixFactory matrixFactory, boolean isTraining) {
		if (isTraining) {
			if (matrix.getColumns() == 1) {
				if (exponentiallyWeightedAverageInputFeatureMeans != null) {
					return exponentiallyWeightedAverageInputFeatureMeans;
				} else {
					return matrix;
				}
			}
		} else {
			if (exponentiallyWeightedAverageInputFeatureMeans != null) {
				return exponentiallyWeightedAverageInputFeatureMeans;
			} else {
				throw new IllegalStateException(
						"Unable to obtain mean for batch " + "norm when not in training - rolling mean not supplied");
			}
		}

		return matrix.rowSums().asEditableMatrix().divi(matrix.getColumns());
	}

	private Matrix getStdDevColumnVector(Matrix varianceColumnVector) {
		EditableMatrix stdDev = varianceColumnVector.dup().asEditableMatrix();
		float epsilion = 0.001f;
		for (int i = 0; i < stdDev.getLength(); i++) {
			float variance = stdDev.get(i);
			float stdDevValue = (float) Math.sqrt(variance + epsilion);
			stdDev.put(i, stdDevValue);
		}
		return stdDev;
	}


	@Override
	protected AxonsType getAxonsType() {
		return AxonsType.createSubType(
				AxonsType.getBaseType(AxonsBaseType.BATCH_NORM), super.getAxonsType().getId());
	}


	@Override
	public Optional<NeuronsActivationFormat<?>> optimisedFor() {
		return axons.optimisedFor();
	}

	@Override
	public boolean isSupported(NeuronsActivationFormat<?> format) {
		return axons.isSupported(format) && NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET.equals(format.getFeatureOrientation());
	}
	

	@Override
	public String toString() {
		return "DefaultBatchNormDirectedAxonsComponentImpl [name='" + name + "', axonsType=" + getAxonsType()
				+ ", inputNeurons=" + getInputNeurons() + ", outputNeurons()=" + getOutputNeurons() + "]";
	}

	@Override
	public Set<DefaultChainableDirectedComponent<?, ?>> flatten() {
		Set<DefaultChainableDirectedComponent<?, ?>> allComponentsIncludingThis = new HashSet<>(Arrays.asList(this));
		return allComponentsIncludingThis;
	}
	
}
