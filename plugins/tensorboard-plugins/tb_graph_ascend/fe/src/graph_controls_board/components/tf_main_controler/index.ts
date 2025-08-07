/* Copyright (c) 2025, Huawei Technologies.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0  (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import '@vaadin/button';
import '@vaadin/combo-box';
import '@vaadin/select';
import { Notification } from '@vaadin/notification';
import { PolymerElement, html } from '@polymer/polymer';
import { customElement, property, observe } from '@polymer/decorators';
import { isEmpty } from 'lodash';
import type { SelectionType } from '../../../graph_ascend/type';
import type { MetaDirType } from '../../type';
@customElement('tf-main-controler')
class MainController extends PolymerElement {
  // 定义模板
  static readonly template = html`
    <style>
      vaadin-text-field input {
        min-height: 0;
      }

      :host {
        --select-border-color: #3b5998;
      }
      vaadin-text-field::part(input-field) {
        background-color: white;
        border: 1px solid #0d0d0d;
        height: 30px;
        border-radius: 0;
        font-size: 14px;
        flex: 1;
      }

      vaadin-combo-box {
        width: 100%;
        padding-top: 10px;
      }
      vaadin-combo-box::part(input-field) {
        height: 30px;
        border: 1px solid var(--paper-input-container-color, var(--secondary-text-color));
        background-color: white;
        font-size: 14px;
        border-radius: 0;
      }
      vaadin-combo-box::part(toggle-button) {
        font-size: 14px;
      }
    </style>

    <div class="control-holder">
      <vaadin-combo-box label="目录" items="[[runs]]" value="{{selectedRun}}"></vaadin-combo-box>
      <vaadin-combo-box label="文件" items="[[tags]]" value="{{selectedTag}}"></vaadin-combo-box>
      <vaadin-combo-box label="MicroStep" items="[[microsteps]]" value="{{selectedMicroStep}}"></vaadin-combo-box>
    </div>
  `;

  @property({ type: Object })
  metaDir: MetaDirType = {};

  @property({ type: Object, notify: true })
  selection: SelectionType = {} as SelectionType;

  @property({ type: Array })
  runs = [];

  @property({ type: Array })
  tags = [];

  @property({ type: Array })
  microsteps = [];

  @property({ type: String })
  selectedRun = '';

  @property({ type: String })
  selectedTag = '';

  @property({ type: Number })
  selectedMicroStep = -1;

  @observe('metaDir')
  _metaDirChanged(): void {
    if (isEmpty(this.metaDir)) {
      return;
    }
    const runs = Object.keys(this.metaDir);
    this.set('runs', runs);
    this.set('selectedRun', runs[0]);
  }

  @observe('selectedRun')
  _selectedRunChanged(): void {
    if (isEmpty(this.metaDir)) {
      return;
    }
    const { type, tags } = this.metaDir[this.selectedRun];
    this.set('tags', tags);
    this.set('selectedTag', tags[0]);
    const selection = {
      ...this.selection,
      run: this.selectedRun,
      tag: tags[0],
      microStep: -1,
      type
    };
    this.set('selectedTag', tags[0]);
    this.set('selectedMicroStep', -1);
    this.set('selection', selection);
  }

  @observe('selectedTag')
  _selectedTagChanged(): void {
    if (isEmpty(this.metaDir)) {
      return;
    }
    const selection = {
      ...this.selection,
      tag: this.selectedTag,
      microStep: -1,
    };
    this.set('selectedMicroStep', -1);
    this.set('selection', selection);
  }

  @observe('selectedMicroStep')
  _selectedMicroStepChanged(): void {
    if (isEmpty(this.metaDir)) {
      return;
    }
    const selection = {
      ...this.selection,
      microStep: this.selectedMicroStep,
    };
    this.set('selection', selection);
  }

  override ready(): void {
    super.ready();
    document.addEventListener('contextMenuTag-changed', this._getTagChanged.bind(this), { passive: true });
  }

  _getTagChanged(event): void {
    const detail = event.detail;
    if (!detail?.rankId || detail?.rankId >= this.tags.length) {
      Notification.show('提示：目标文件不存在', {
        position: 'middle',
        duration: 2000,
        theme: 'warning',
      });
      return;
    }
    this.set('selectedTag', this.tags[detail?.rankId]);
  }
}
