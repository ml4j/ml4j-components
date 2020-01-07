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
package org.ml4j.nn.activationfunctions;

import org.ml4j.nn.activationfunctions.base.DifferentiableActivationFunctionActivationBase;
import org.ml4j.nn.neurons.NeuronsActivation;

/**
 * Default implementation of a DifferentiableActivationFunctionActivation.
 * 
 * Encapsulates the activations from an activation through a DifferentiableActivationFunctionComponent
 * 
 * @author Michael Lavelle
 */
public class DefaultDifferentiableActivationFunctionActivationImpl extends DifferentiableActivationFunctionActivationBase implements DifferentiableActivationFunctionActivation {

	public DefaultDifferentiableActivationFunctionActivationImpl(DifferentiableActivationFunction activationFunction,
			NeuronsActivation input, NeuronsActivation output) {
		super(activationFunction, input, output);
	}
}
