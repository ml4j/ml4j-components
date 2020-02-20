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

import java.util.List;
import java.util.Optional;
import java.util.function.Supplier;

import org.ml4j.EditableMatrix;
import org.ml4j.InterrimMatrix;
import org.ml4j.Matrix;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.AxonsGradient;
import org.ml4j.nn.axons.AxonsGradientImpl;
import org.ml4j.nn.axons.TrainableAxons;
import org.ml4j.nn.components.DirectedComponentActivationLifecycle;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.axons.base.DirectedAxonsComponentActivationBase;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of DirectedAxonsComponentActivation.
 * 
 * Encapsulates the activations from a forward propagation through a
 * DirectedAxonsComponent
 * 
 * @author Michael Lavelle
 * 
 * @param <A> The type of Axons within the DirectedAxonsComponent from which
 *            this activation originated
 */
public class DefaultDirectedAxonsComponentActivationImpl<A extends Axons<?, ?, ?>>
		extends DirectedAxonsComponentActivationBase<A> implements DirectedAxonsComponentActivation {

	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultDirectedAxonsComponentActivationImpl.class);

	public DefaultDirectedAxonsComponentActivationImpl(DirectedAxonsComponent<?, ?, A> axonsComponent,
			AxonsActivation axonsActivation, AxonsContext axonsContext) {
		super(axonsComponent, axonsActivation, axonsContext);
	}

	@Override
	public float getTotalRegularisationCost() {
		float totalRegularisationCost = 0f;
		if (axonsContext.getRegularisationLambda() != 0) {

			LOGGER.debug("Calculating total regularisation cost");

			if (directedAxonsComponent.getAxons() instanceof TrainableAxons) {

				try (InterrimMatrix weightsWithoutBiases = ((TrainableAxons<?, ?, ?>) directedAxonsComponent.getAxons())
						.getDetachedAxonWeights().getConnectionWeights().getMatrix().asInterrimMatrix()) {
					float regularisationMatrix = weightsWithoutBiases.asEditableMatrix().muli(weightsWithoutBiases)
							.sum();
					totalRegularisationCost = totalRegularisationCost
							+ ((axonsContext.getRegularisationLambda()) * regularisationMatrix) / 2;
				}
			}
		}
		return totalRegularisationCost;
	}

	@Override
	protected DirectedComponentGradientImpl<NeuronsActivation> createBackPropagatedGradient(
			AxonsActivation axonsActivation, List<Supplier<AxonsGradient>> previousAxonsGradients,
			Supplier<AxonsGradient> thisAxonsGradient) {
		return new DirectedComponentGradientImpl<>(previousAxonsGradients, thisAxonsGradient,
				axonsActivation.getPostDropoutOutput());
	}

	@Override
	protected Optional<AxonsGradient> getCalculatedAxonsGradient(AxonsActivation rightToLeftAxonsGradientActivatoin) {

		if (directedAxonsComponent.getAxons().isTrainable(axonsContext)) {

			TrainableAxons<?, ?, ?> trainableAxons = (TrainableAxons<?, ?, ?>) directedAxonsComponent.getAxons();

			LOGGER.debug("Calculating Axons Gradients");
			NeuronsActivation rightToLeftPostDropoutInput = rightToLeftAxonsGradientActivatoin.getPostDropoutInput()
					.get();

			Matrix first = rightToLeftPostDropoutInput.getActivations(axonsContext.getMatrixFactory());
			
			EditableMatrix totalTrainableAxonsGradientMatrixNonBias = null;
			Matrix totalTrainableAxonsGradientMatrixBias = null;

			NeuronsActivation leftToRightPostDropoutInputActivation = leftToRightAxonsActivation.getPostDropoutInput()
					.get();

			try (InterrimMatrix leftToRightPostDropoutInputActivationMatrix = leftToRightPostDropoutInputActivation
					.getActivations(axonsContext.getMatrixFactory()).asInterrimMatrix()) {
				try (InterrimMatrix second = leftToRightPostDropoutInputActivationMatrix.transpose()
						.asInterrimMatrix()) {

					totalTrainableAxonsGradientMatrixNonBias = first.mmul(second).asEditableMatrix();
					leftToRightPostDropoutInputActivation.close();
				}
			}

			if (directedAxonsComponent.getAxons().getLeftNeurons().hasBiasUnit()) {
				totalTrainableAxonsGradientMatrixBias = first.rowSums();
			}

			if (axonsContext.getRegularisationLambda() != 0) {

				LOGGER.debug("Calculating total regularisation Gradients");

				try (InterrimMatrix connectionWeightsCopy = trainableAxons.getDetachedAxonWeights()
						.getConnectionWeights().getMatrix().asInterrimMatrix()) {

					Matrix regularisationAddition = connectionWeightsCopy.asEditableMatrix()
							.muli(axonsContext.getRegularisationLambda());

					totalTrainableAxonsGradientMatrixNonBias.addi(regularisationAddition);

				}
			}

			rightToLeftPostDropoutInput.close();

			return Optional.of(new AxonsGradientImpl((TrainableAxons<?, ?, ?>) directedAxonsComponent.getAxons(),
					totalTrainableAxonsGradientMatrixNonBias, totalTrainableAxonsGradientMatrixBias));

		} else {
			NeuronsActivation postDropoutInput = rightToLeftAxonsGradientActivatoin.getPostDropoutInput().get();
			NeuronsActivation leftToRightPostDropoutInputActivation = leftToRightAxonsActivation.getPostDropoutInput()
					.get();
			if (postDropoutInput != null) {
				postDropoutInput.close();
			}
			if (leftToRightPostDropoutInputActivation != null) {
				postDropoutInput.close();
			}
			postDropoutInput.close();
			return Optional.empty();
		}
	}

	private void close(NeuronsActivation activation) {
		if (!activation.isImmutable()) {
			activation.close();
		}
	}

	@Override
	public void close(DirectedComponentActivationLifecycle completedLifeCycleStage) {
		if (completedLifeCycleStage == DirectedComponentActivationLifecycle.FORWARD_PROPAGATION) {
			close(getOutput());
			close(leftToRightAxonsActivation.getPostDropoutOutput());
		}
	}
}
