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
export interface HierarchyNodeType {
    x: number;
    y: number;
    width: number;
    height: number;
    expand: boolean;
    parentNode: string;
    label: string;
    name: string;
    isRoot: boolean;
    children?: HierarchyNodeType[];
    nodeType: number;
    matchedNodeLink: string[];
    matchedDistributed: { communications_type: 'send' | 'receive'; nodes_info: { [string]: Array<string> } };
    precisionIndex: string;
    overflowLevel: string;
};

export interface ContextMenuItem {
    text?: string;
    rankId?: number;
    component?: any;
    nodeName?: string;
    type?: number;
    children?: ContextMenuItem[];
}

export interface PreProcessDataConfigType {
    colors: { string: { value: number[]; color: string } };
    isOverflowFilter: boolean;
    graphType: GraphType;
}

export type GraphType = 'NPU' | 'Bench' | 'Single';

export interface UseGraphType {
    bindInnerRect: (container: any, data: any) => void;
    bindOuterRect: (container: any, data: any) => void;
    bindText: (container: any, data: any) => void;
    preProcessData: (
        hierarchyObject: { [key: string]: HierarchyNodeType },
        data: any[],
        selectedNode: string,
        config: PreProcessDataConfigType,
        transform: { x: number; y: number; scale: number },
    ) => Array<any>;
    changeNodeExpandState: (nodeInfo: any, metaData: any) => Promise<any>;
    createComponent: (text, precision, colors: PreProcessDataConfigType['colors']) => any;
    updateHierarchyData: (graphType: string) => Promise<any>;
}

export interface TransformType {
    x: number;
    y: number;
    scale: number;
};
