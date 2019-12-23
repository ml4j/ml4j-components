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

import java.util.function.Supplier;

import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.AxonsGradient;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.axons.base.DirectedAxonsComponentActivationBase;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DefaultBatchNormDirectedAxonsComponentActivationImpl extends DirectedAxonsComponentActivationBase implements DirectedAxonsComponentActivation {
	
	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultBatchNormDirectedAxonsComponentActivationImpl.class);

	private AxonsContext axonsContext;
	private AxonsActivation leftToRightAxonsActivation;
	
	public DefaultBatchNormDirectedAxonsComponentActivationImpl(AxonsContext axonsContext, DirectedAxonsComponent<?, ?> axonsComponent, AxonsActivation axonsActivation) {
		super(axonsComponent, axonsActivation.getPostDropoutOutput());
		this.axonsContext = axonsContext;
		this.leftToRightAxonsActivation = axonsActivation;
	}
	
	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> gradient) {
		LOGGER.debug("Back propagating gradient through DefaultBatchNormDirectedAxonsComponentActivationImpl");
		AxonsActivation rightToLeftAxonsActivation = directedAxonsComponent.getAxons().pushRightToLeft(gradient.getOutput(), leftToRightAxonsActivation, axonsContext);
		return new DirectedComponentGradientImpl<NeuronsActivation>(gradient.getTotalTrainableAxonsGradients(), getCalculatedAxonsGradient(rightToLeftAxonsActivation), rightToLeftAxonsActivation.getPostDropoutOutput());
	}
	
	private Supplier<AxonsGradient> getCalculatedAxonsGradient(AxonsActivation inboundGradientActivation) {
		return () -> null;
	}

	@Override
	public float getTotalRegularisationCost() {
		return 0;
	}
}
