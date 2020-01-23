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
package org.ml4j.nn.components.activationfunctions;

import org.ml4j.InterrimMatrix;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.components.DirectedComponentActivationLifecycle;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.activationfunctions.base.DifferentiableActivationFunctionComponentActivationBase;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of
 * DifferentiableActivationFunctionComponentActivation, encapsulating the
 * activations from a DifferentiableActivationFunctionComponent and providing
 * the logic required in order to back propagate gradients back through the
 * activations.
 * 
 * @author Michael Lavelle
 */
public class DefaultDifferentiableActivationFunctionComponentActivationImpl extends
		DifferentiableActivationFunctionComponentActivationBase<DifferentiableActivationFunctionComponentAdapter>
		implements DifferentiableActivationFunctionComponentActivation {

	private static final Logger LOGGER = LoggerFactory
			.getLogger(DefaultDifferentiableActivationFunctionComponentActivationImpl.class);

	private DifferentiableActivationFunctionActivation activationFunctionActivation;
	private NeuronsActivationContext activationContext;

	/**
	 * @param activationFunctionComponent  The
	 *                                     DifferentiableActivationFunctionComponent
	 *                                     that generated this activation.
	 * @param activationFunctionActivation The activation from the underlying
	 *                                     DifferentiableActivationFunction.
	 * @param activationContext            The activation context.
	 */
	public DefaultDifferentiableActivationFunctionComponentActivationImpl(
			DifferentiableActivationFunctionComponentAdapter activationFunctionComponent,
			DifferentiableActivationFunctionActivation activationFunctionActivation,
			NeuronsActivationContext activationContext) {
		super(activationFunctionComponent, activationFunctionActivation.getInput(),
				activationFunctionActivation.getOutput());
		this.activationFunctionActivation = activationFunctionActivation;
		this.activationContext = activationContext;
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> gradient) {
		LOGGER.debug("Back propagating gradient through DifferentiableActivationFunctionComponentActivation");
		NeuronsActivation backPropagatedGradient = originatingComponent.getActivationFunction()
				.activationGradient(activationFunctionActivation, activationContext);
		try (InterrimMatrix backPropGradientMatrix = backPropagatedGradient
				.getActivations(activationContext.getMatrixFactory()).asInterrimMatrix()) {
			DirectedComponentGradient<NeuronsActivation> result = new DirectedComponentGradientImpl<NeuronsActivation>(
					gradient.getTotalTrainableAxonsGradients(),
					new NeuronsActivationImpl(gradient.getOutput().getNeurons(),
							backPropGradientMatrix.asEditableMatrix()
									.mul(gradient.getOutput().getActivations(activationContext.getMatrixFactory())),
									gradient.getOutput().getFormat()));

			activationFunctionActivation.getInput().close();

			if (!gradient.getOutput().isImmutable()) {
				gradient.getOutput().close();
			}

			return result;

		}
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(CostFunctionGradient costFunctionGradient) {
		LOGGER.debug(
				"Back propagating cost function gradient through DifferentiableActivationFunctionComponentActivation");
		return costFunctionGradient.backPropagateThroughFinalActivationFunction(
				originatingComponent.getActivationFunction().getActivationFunctionType());
	}

	@Override
	public void close(DirectedComponentActivationLifecycle completedLifeCycleStage) {
		if (completedLifeCycleStage == DirectedComponentActivationLifecycle.FORWARD_PROPAGATION) {
			if (!activationFunctionActivation.getOutput().isImmutable()) {
				activationFunctionActivation.getOutput().close();
			}
		}
	}
}
