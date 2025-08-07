/* -------------------------------------------------------------------------
 Copyright (c) 2025, Huawei Technologies.
 All rights reserved.

 Licensed under the Apache License, Version 2.0  (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
--------------------------------------------------------------------------------------------*/
const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
    mode: 'development', // 明确指定开发模式
    devtool: 'eval-cheap-source-map', // 开发环境推荐使用的 source map 类型

    entry: {
        app: './src/index', // 保持与生产环境一致的入口
    },

    output: {
        filename: '[name].bundle.js', // 使用带 hash 的文件名
        path: path.resolve(__dirname, 'dist'),
        publicPath: '/', // 确保 dev server 的静态资源路径正确
    },

    devServer: {
        static: {
            directory: path.join(__dirname, 'dist'), // 服务资源目录
        },
        compress: false,
        proxy: [
            {
                context: (pathname) => {
                    return !pathname.match(/(\.js|\.css|\.html|\.ico|\.svg)$/);
                },
                target: 'http://127.0.0.1:6006',
                changeOrigin: true,
                secure: false,
                pathRewrite: {
                    '^/(.*)': '/data/plugin/graph_ascend/$1', // 路径转换核心逻辑
                },
                on: {
                    error: (err, req, res) => {
                        // 安全处理响应对象
                        if (res && !res.headersSent) {
                            res.writeHead(500, { 'Content-Type': 'text/plain' });
                            res.end('Proxy Error');
                        }
                    },
                    proxyReqWs: (proxyReq, req, socket) => {
                        // WebSocket 错误专属处理
                        socket.on('error', (error) => {

                        });
                    },
                },
            },
        ],
        webSocketServer: {
            type: 'ws',
            options: {
                path: '/ws',
                noInfo: true,
            },
        },
        http2: false, // 推荐启用HTTP/2
        https: false, // 根据实际需要配置
        hot: true, // 启用热模块替换
        liveReload: true, // 启用实时重新加载
        port: 8080, // 自定义端口号
        open: true, // 自动打开浏览器
        client: {
            overlay: {
                errors: true,
                warnings: false,
            }, // 在浏览器中显示编译错误
        },
        headers: {
            'X-Proxy-Source': 'webpack-dev-server',
        },
    },

    module: {
        rules: [
            {
                test: /\.html$/,
                use: [
                    {
                        loader: 'html-loader',
                        options: {
                            sources: false, // 开发环境不需要优化资源路径
                        },
                    },
                ],
            },
            {
                test: /\.ts?$/,
                use: {
                    loader: 'ts-loader',
                    options: {
                        transpileOnly: true, // 保持快速编译
                        experimentalWatchApi: true, // 启用 TypeScript 的监听 API
                    },
                },
                exclude: /node_modules/,
            },
            {
                test: /\.css$/i,
                use: [
                    'style-loader',
                    {
                        loader: 'css-loader',
                        options: {
                            sourceMap: true, // 启用 CSS source maps
                        },
                    },
                ],
            },
        ],
    },

    resolve: {
        extensions: ['.ts', '.js', '.json'], // 添加 .json 扩展名解析
    },

    plugins: [
        new HtmlWebpackPlugin({
            inject: 'body',
            template: './index.html',
            minify: false, // 开发环境不压缩 HTML
        }),
    ],

    optimization: {
        removeAvailableModules: false,
        removeEmptyChunks: false,
        splitChunks: false, // 禁用代码拆分加速编译
    },
};