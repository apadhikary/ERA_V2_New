"use strict";(globalThis.webpackChunk=globalThis.webpackChunk||[]).push([["codespaces"],{61013:(e,t,n)=>{n(83058);var r=n(76006),o=n(69567),a=n(12040),i=n(59753),s=n(5223);function l(e){let t=e.querySelectorAll(".js-toggle-hidden");for(let e of t)e.hidden=!e.hidden;let n=e.querySelectorAll(".js-toggle-disabled");for(let e of n)e.getAttribute("aria-disabled")?e.removeAttribute("aria-disabled"):e.setAttribute("aria-disabled","true")}async function c(){let e=document.querySelector(".js-codespaces-details-container");e&&(e.open=!1);let t=document.querySelector("new-codespace");if(t&&!t.getAttribute("data-no-submit-on-create"))try{let e=await fetch("/codespaces/new");if(e&&e.ok){let n=(0,s.r)(document,await e.text());t.replaceWith(n)}}catch(e){}}async function u(e,t){let n=document.querySelector(`#${e}`),r=await (0,a.W)({content:n.content.cloneNode(!0),dialogClass:"project-dialog"});return t&&t.setAttribute("aria-expanded","true"),r.addEventListener("dialog:remove",function(){t&&l(t)},{once:!0}),r}async function d(e){let t=await fetch(e.action,{method:e.method,body:new FormData(e),headers:{Accept:"application/json","X-Requested-With":"XMLHttpRequest"}});if(t.ok){let n=await t.json();if(n.codespace_url)window.location.href=n.codespace_url,l(e),c(),f();else{let t=e.closest("get-repo")||e.closest("new-codespace");t?(e.setAttribute("data-src",n.loading_url),e.dispatchEvent(new CustomEvent("pollvscode"))):e.closest("create-button")&&(e.setAttribute("data-src",n.loading_url),e.dispatchEvent(new CustomEvent("prpollvscode"))),l(e)}}else if(422===t.status){let n=await t.json();if("concurrency_limit_error"===n.error_type)await u("concurrency-error",e);else{let t=document.querySelector("template.js-flash-template"),r=n.error;t.after(new o.R(t,{className:"flash-error",message:r})),l(e)}}}async function f(){let e=document.querySelector(".js-codespaces-completable"),t=e&&e.getAttribute("data-src");if(!t)return;let n=await fetch(t,{method:"GET",headers:{Accept:"text/fragment+html","X-Requested-With":"XMLHttpRequest"}});if(n.ok){let t=(0,s.r)(document,await n.text());e.replaceWith(t)}else throw Error(`Unexpected response: ${n.statusText}`)}(0,i.on)("submit",".js-toggle-hidden-codespace-form",function(e){let t=e.currentTarget;l(t)}),(0,i.on)("submit",".js-create-codespaces-form-command",function(e){let t=e.currentTarget;t.classList.contains("js-open-in-vscode-form")||(c(),l(t))}),(0,i.on)("submit","form.js-open-in-vscode-form",async function(e){e.preventDefault();let t=e.currentTarget;await d(t)});let m=class ConcurrencyLimitElement extends HTMLElement{async connectedCallback(){u("concurrency-error")}};m=function(e,t,n,r){var o,a=arguments.length,i=a<3?t:null===r?r=Object.getOwnPropertyDescriptor(t,n):r;if("object"==typeof Reflect&&"function"==typeof Reflect.decorate)i=Reflect.decorate(e,t,n,r);else for(var s=e.length-1;s>=0;s--)(o=e[s])&&(i=(a<3?o(i):a>3?o(t,n,i):o(t,n))||i);return a>3&&i&&Object.defineProperty(t,n,i),i}([r.Ih],m);var p=n(80702),h=n(226);function y(e,t,n,r){var o,a=arguments.length,i=a<3?t:null===r?r=Object.getOwnPropertyDescriptor(t,n):r;if("object"==typeof Reflect&&"function"==typeof Reflect.decorate)i=Reflect.decorate(e,t,n,r);else for(var s=e.length-1;s>=0;s--)(o=e[s])&&(i=(a<3?o(i):a>3?o(t,n,i):o(t,n))||i);return a>3&&i&&Object.defineProperty(t,n,i),i}let g=class NewCodespaceElement extends HTMLElement{async connectedCallback(){let e=new URL(document.location.href,window.location.origin),t=new URLSearchParams(e.search);t.has("response_error")&&(t.delete("response_error"),window.history.replaceState({},"",`?${t.toString()}`))}toggleLoadingVscode(){let e=this.loadingVscode.hidden,t=this.children;for(let n=0;n<t.length;n++)t[n].hidden=e;this.loadingVscode.hidden=!e}machineTypeSelected(e){let t=e.currentTarget,n=t.getAttribute("data-sku-name");this.skuNameInput&&n&&(this.skuNameInput.value=n),this.advancedOptionsForm&&(0,p.Bt)(this.advancedOptionsForm)}pollForVscode(e){this.toggleLoadingVscode();let t=e.currentTarget.getAttribute("data-src");t&&this.vscodePoller.setAttribute("src",t)}vscsTargetUrlUpdated(e){let t=e.currentTarget;this.vscsTargetUrl.value=t.value}async declarativeSecretsHashUpdated(e){let t=e.currentTarget,n=t.getAttribute("data-name");if(!n)return;let r=t.value,o=(0,h.Jx)(t.getAttribute("data-public-key"));"checkbox"!==t.getAttribute("type")||t.checked||(r=""),r?this.secrets_hash.set(n,(0,h.cv)(await (0,h.HI)(o,r))):this.secrets_hash.delete(n),this.declarativeSecretsHash.value=JSON.stringify(Object.fromEntries(this.secrets_hash))}constructor(...e){super(...e),this.secrets_hash=new Map}};function v(e,t,n,r){var o,a=arguments.length,i=a<3?t:null===r?r=Object.getOwnPropertyDescriptor(t,n):r;if("object"==typeof Reflect&&"function"==typeof Reflect.decorate)i=Reflect.decorate(e,t,n,r);else for(var s=e.length-1;s>=0;s--)(o=e[s])&&(i=(a<3?o(i):a>3?o(t,n,i):o(t,n))||i);return a>3&&i&&Object.defineProperty(t,n,i),i}y([r.fA],g.prototype,"declarativeSecretsHash",void 0),y([r.fA],g.prototype,"vscsTargetUrl",void 0),y([r.fA],g.prototype,"loadingVscode",void 0),y([r.fA],g.prototype,"vscodePoller",void 0),y([r.fA],g.prototype,"advancedOptionsForm",void 0),y([r.fA],g.prototype,"skuNameInput",void 0),g=y([r.Ih],g);let b=class ExportBranchElement extends HTMLElement{connectedCallback(){this.abortPoll=new AbortController,this.loadingIndicator.hidden||this.startPoll()}disconnectedCallback(){this.abortPoll?.abort()}applyPublishParams(){let e=this.form.getAttribute("data-codespace-id"),t=document.querySelector(`[data-codespace-id='${e}'][data-class="publish-codespace-form"]`);if(t){let e=Object.fromEntries(new FormData(t).entries());if(this.form){let t=`?name=${e.name}&visibility=${e.visibility}`,n=(this.form.getAttribute("action")||"").split("?")[0]+t;this.form.setAttribute("action",n)}}}async exportBranch(e){e.preventDefault(),this.form.hidden=!0,this.loadingIndicator.hidden=!1;let t=await fetch(this.form.action,{method:this.form.method,body:new FormData(this.form),headers:{Accept:"text/fragment+html","X-Requested-With":"XMLHttpRequest"}});t.ok?this.startPoll():(this.form.hidden=!1,this.loadingIndicator.hidden=!0)}async startPoll(){let e=this.getAttribute("data-exported-codespace-url")||"",t=await this.poll(e);if(t){if(t.ok)this.loadingIndicator.hidden=!0,this.viewBranchLink.hidden=!1;else{let e=this.getAttribute("data-export-error-redirect-url")||"";window.location.href=encodeURI(e)}}}async poll(e,t=1e3){if(this.abortPoll?.signal.aborted)return;let n=await fetch(e,{signal:this.abortPoll?.signal});return 202===n.status||404===n.status?(await new Promise(e=>setTimeout(e,t)),this.poll(e,1.5*t)):n}constructor(...e){super(...e),this.abortPoll=null}};function w(e,t,n,r){var o,a=arguments.length,i=a<3?t:null===r?r=Object.getOwnPropertyDescriptor(t,n):r;if("object"==typeof Reflect&&"function"==typeof Reflect.decorate)i=Reflect.decorate(e,t,n,r);else for(var s=e.length-1;s>=0;s--)(o=e[s])&&(i=(a<3?o(i):a>3?o(t,n,i):o(t,n))||i);return a>3&&i&&Object.defineProperty(t,n,i),i}v([r.fA],b.prototype,"form",void 0),v([r.fA],b.prototype,"loadingIndicator",void 0),v([r.fA],b.prototype,"viewBranchLink",void 0),b=v([r.Ih],b),n(80583),n(82735);let S=class CodespaceZeroConfigElement extends HTMLElement{connectedCallback(){this.toggleLocationConfigs("production")}updateVscsTargets(){for(let e of(this.vscsTargetUrl.disabled="local"!==this.vscsTarget.value,this.toggleLocationConfigs(this.vscsTarget.value),this.vscsTargets))e.value=this.vscsTarget.value}updateVscsTargetUrls(){for(let e of this.vscsTargetUrls)e.value=this.vscsTargetUrl.value}updateLocations(e){let t=e.currentTarget;this.setLocationValues(t.value)}setLocationValues(e){for(let t of this.locations)t.value=e}toggleLocationConfigs(e){for(let t of this.locationConfigs)if(t.getAttribute("data-vscs-target")===e){t.hidden=!1;let e=t.querySelector("option");e&&(e.selected=!0,this.setLocationValues(e.value))}else t.hidden=!0}};w([r.fA],S.prototype,"regionConfig",void 0),w([r.fA],S.prototype,"vscsTarget",void 0),w([r.fA],S.prototype,"vscsTargetUrl",void 0),w([r.GO],S.prototype,"locationConfigs",void 0),w([r.GO],S.prototype,"vscsTargets",void 0),w([r.GO],S.prototype,"vscsTargetUrls",void 0),w([r.GO],S.prototype,"locations",void 0),S=w([r.Ih],S)},226:(e,t,n)=>{n.d(t,{HI:()=>a,Jx:()=>i,cv:()=>s});var r=n(69567),o=n(59753);async function a(e,t){let r=new TextEncoder,o=r.encode(t),{seal:a}=await Promise.all([n.e("vendors-node_modules_blakejs_index_js-node_modules_tweetnacl_nacl-fast_js"),n.e("_empty-file_js-app_assets_modules_github_tweetsodium_ts")]).then(n.bind(n,99640));return a(o,e)}function i(e){let t=atob(e).split("").map(e=>e.charCodeAt(0));return Uint8Array.from(t)}function s(e){let t="";for(let n of e)t+=String.fromCharCode(n);return btoa(t)}function l(e){return async function(t){let n=t.currentTarget;if(t.defaultPrevented||!n.checkValidity())return;let o=i(n.getAttribute("data-public-key"));for(let i of(t.preventDefault(),n.elements))if(i.id.endsWith("secret")){if(i.disabled=!0,i.required&&!i.value){let e=`${i.name} is invalid!`,t=document.querySelector("template.js-flash-template");t.after(new r.R(t,{className:"flash-error",message:e}));return}let t=`${i.name}_encrypted_value`;if(!i.value){n.elements.namedItem(t).disabled=e;continue}n.elements.namedItem(t).value=s(await a(o,i.value))}n.submit()}}(0,o.on)("submit","form.js-encrypt-submit",async function(e){let t=e.currentTarget;if(e.defaultPrevented||!t.checkValidity())return;let n=t.elements.namedItem("secret_value");if(n.disabled=!0,!n.value)return;e.preventDefault();let r=i(t.getAttribute("data-public-key"));t.elements.namedItem("encrypted_value").value=s(await a(r,n.value)),t.submit()}),(0,o.on)("submit","form.js-encrypt-bulk-submit",l(!0)),(0,o.on)("submit","form.js-encrypt-bulk-submit-enable-empty",l(!1))},83058:(e,t,n)=>{n.d(t,{a:()=>s});var r=n(36071),o=n(59753),a=n(65935),i=n(80702);function s(e,t){let n;let r=t.querySelector("*"),o=e.ownerDocument.activeElement;o instanceof HTMLElement&&(n=r?.querySelector(function(e){let t=e.tagName.toLowerCase(),n=e.hasAttribute("class")?`.${e.className.split(" ").join(".")}`:"",r=e.hasAttribute("id")?`#${e.id}`:"",o=e.hasAttribute("name")?`[name="${e.getAttribute("name")}"]`:"";return`${t}${r}${n}${o}`}(o))),e.replaceWith(t),n instanceof HTMLElement&&n.focus()}function l(){let e=new URL(document.location.href,window.location.origin),t=new URLSearchParams(e.search);t.set("response_error","true"),window.location.replace(`${window.location.pathname}?${t.toString()}`)}(0,o.on)("remote-input-error","#js-codespaces-repository-select",()=>{let e=document.querySelector("#js-codespaces-unable-load-repositories-warning");e.hidden=!1}),(0,a.AC)(".js-new-codespace-form",async function(e,t){let n=e.closest("[data-replace-remote-form-target]"),r=n.querySelector(".js-new-codespace-submit-button");r instanceof HTMLInputElement&&(r.disabled=!0),e.classList.remove("is-error"),e.classList.add("is-loading");try{r&&r.setAttribute("disabled","true");let e=await t.html();if(200!==e.status&&l(),s(n,e.html),"true"===n.getAttribute("data-allow-update-url")){let e=new FormData(document.querySelector("form.js-new-codespace-form"));!function(e){let t=new URL(document.location.href,window.location.origin),n=new URLSearchParams(t.search),r=["vscs_target"];for(let[t,o]of e.entries()){if(r.includes(t)||!o){n.delete(t);continue}n.set(t,o)}window.history.replaceState({},"",`?${n.toString()}`)}(e)}}catch(e){throw l(),e}});let c=null;function u(e){if(c=e,null!==e){let e=document.querySelector(".js-codespace-loading-steps");e.setAttribute("data-current-state",c)}}(0,r.N7)(".js-codespace-loading-steps",{constructor:HTMLElement,add:e=>{let t=e.getAttribute("data-current-state");t&&u(t)}}),(0,r.N7)(".js-codespace-advance-state",{constructor:HTMLElement,add:e=>{let t=e.getAttribute("data-state");t&&u(t)}});let d=null;function f(e){let t=document.querySelector(".js-workbench-form-container form");t&&e?(function(e,t){let n=e.querySelector(".js-cascade-token");n&&n.setAttribute("value",t)}(t,e),function(e,t){let n=e.querySelector(".js-partner-info");if(n){let e=n.getAttribute("value");e&&(e=e.replace("%CASCADE_TOKEN_PLACEHOLDER%",t),n.setAttribute("value",e))}}(t,e),(0,i.Bt)(t)):u("failed")}(0,r.N7)(".js-auto-submit-form",{constructor:HTMLFormElement,initialize:i.Bt}),(0,r.N7)(".js-workbench-form-container",{constructor:HTMLElement,add:e=>{let t=e.querySelector(".js-cascade-token");(function(e,t){if(""!==t.value){let t=e.querySelector("form");(0,i.Bt)(t)}else{let e=document.querySelector(".js-fetch-cascade-token");(0,a.AC)(".js-fetch-cascade-token",async function(e,t){try{let e=await t.json(),n=e.json;d=n.token}catch(e){}}),(0,i.Bt)(e),function(e,t,n){let r=document.querySelector(e);if(r){let e=Date.now(),n=setInterval(()=>{let r=Date.now()-e;if(d||r>=1e4){clearInterval(n),t(d||void 0);return}},50)}}(".js-workbench-form-container",f,0)}})(e,t)}})},80583:(e,t,n)=>{var r=n(76006);function o(e,t,n,r){var o,a=arguments.length,i=a<3?t:null===r?r=Object.getOwnPropertyDescriptor(t,n):r;if("object"==typeof Reflect&&"function"==typeof Reflect.decorate)i=Reflect.decorate(e,t,n,r);else for(var s=e.length-1;s>=0;s--)(o=e[s])&&(i=(a<3?o(i):a>3?o(t,n,i):o(t,n))||i);return a>3&&i&&Object.defineProperty(t,n,i),i}let a=class InputDemuxElement extends HTMLElement{connectedCallback(){this.control&&(this.storedInput=Array(this.control.children.length).fill("")),this.addEventListener("input",this.relayInput.bind(this)),this.addEventListener("keydown",this.relayKeydown.bind(this));let e=this.closest("details");e&&e.addEventListener("toggle",()=>{e.open&&this.source.focus()})}relayKeydown(e){if((this.isControlTab(e.target)||e.target===this.source)&&("ArrowDown"===e.key||"Tab"===e.key))e.preventDefault(),e.stopPropagation(),this.routeCustomEvent(new CustomEvent("focus-list"));else if("Escape"===e.key){let e=this.closest("details");e&&e.removeAttribute("open")}}isControlTab(e){return!!e&&!!this.control&&Array.from(this.control.children).includes(e)}relayInput(e){if(!e.target)return;let t=e.target,n=t.value;this.routeCustomEvent(new CustomEvent("input-entered",{detail:n}))}routeCustomEvent(e){let t=this.sinks[this.selectedIndex];t.dispatchEvent(e)}get selectedIndex(){if(!this.control)return 0;let e=this.control.querySelector('[aria-selected="true"]');return e?Array.from(this.control.children).indexOf(e):0}storeInput(){this.storedInput[this.selectedIndex]=this.source.value}updateInput(e){this.source.value=this.storedInput[this.selectedIndex];let t=e.detail.relatedTarget.getAttribute("data-filter-placeholder");this.source.placeholder=t,this.source.setAttribute("aria-label",t),this.notifySelected()}notifySelected(){let e=this.sinks[this.selectedIndex],t=new CustomEvent("tab-selected");e.dispatchEvent(t)}};o([r.fA],a.prototype,"source",void 0),o([r.GO],a.prototype,"sinks",void 0),o([r.fA],a.prototype,"control",void 0),a=o([r.Ih],a)},40578:(e,t,n)=>{let r;function o(){if(!r)throw Error("Client env was requested before it was loaded. This likely means you are attempting to use client env at the module level in SSR, which is not supported. Please move your client env usage into a function.");return r}function a(){return r?.locale??"en-US"}n.d(t,{Kd:()=>a,dU:()=>o}),function(){if("undefined"!=typeof document){let e=document.getElementById("client-env");if(e)try{r=JSON.parse(e.textContent||"")}catch(e){console.error("Error parsing client-env",e)}}}()},12040:(e,t,n)=>{n.d(t,{W:()=>o});var r=n(59753);async function o(e){let t=document.querySelector("#site-details-dialog"),n=t.content.cloneNode(!0),o=n.querySelector("details"),a=o.querySelector("details-dialog"),i=o.querySelector(".js-details-dialog-spinner");e.detailsClass&&o.classList.add(...e.detailsClass.split(" ")),e.dialogClass&&a.classList.add(...e.dialogClass.split(" ")),e.label?a.setAttribute("aria-label",e.label):e.labelledBy&&a.setAttribute("aria-labelledby",e.labelledBy),document.body.append(n);try{let t=await e.content;i.remove(),a.prepend(t)}catch(n){i.remove();let t=document.createElement("span");t.textContent=e.errorMessage||"Couldn't load the content",t.classList.add("my-6"),t.classList.add("mx-4"),a.prepend(t)}return o.addEventListener("toggle",()=>{o.hasAttribute("open")||((0,r.f)(a,"dialog:remove"),o.remove())}),a}},87274:(e,t,n)=>{n.d(t,{C:()=>a,x:()=>o});var r=n(71207);let o=r.n4?.readyState==="interactive"||r.n4?.readyState==="complete"?Promise.resolve():new Promise(e=>{r.n4?.addEventListener("DOMContentLoaded",()=>{e()})}),a=r.n4?.readyState==="complete"?Promise.resolve():new Promise(e=>{r.iG?.addEventListener("load",e)})},65809:(e,t,n)=>{n.d(t,{eK:()=>y});var r=n(82918),o=n(21630),a=n(28382);function i(e){let t=document.querySelectorAll(e);if(t.length>0)return t[t.length-1]}var s=n(54508),l=n(71207),c=n(18515),u=n(58374);let d=!1,f=0,m=Date.now(),p=new Set(["Failed to fetch","NetworkError when attempting to fetch resource."]);function h(e){return!!("AbortError"===e.name||"TypeError"===e.name&&p.has(e.message)||e.name.startsWith("ApiError")&&p.has(e.message))}function y(e,t={}){if((0,c.c)("FAILBOT_HANDLE_NON_ERRORS")){if(!(e instanceof Error||"object"==typeof e&&null!==e&&"name"in e&&"string"==typeof e.name&&"message"in e&&"string"==typeof e.message)){if(function(e){if(!e||"boolean"==typeof e||"number"==typeof e)return!0;if("string"==typeof e){if(E.some(t=>e.includes(t)))return!0}else if("object"==typeof e&&"string"==typeof e.message&&"number"==typeof e.code)return!0;return!1}(e))return;let n=Error(),r=function(e){try{return JSON.stringify(e)}catch{return"Unserializable"}}(e),o={type:"UnknownError",value:`Unable to report error, due to a thrown non-Error type: ${typeof e}, with value ${r}`,stacktrace:w(n),catalogService:void 0};g(b(o,t));return}h(e)||g(b(v(e),t))}else h(e)||g(b(v(e),t))}async function g(e){if(!(!_&&!d&&f<10&&(0,o.Gb)()))return;let t=document.head?.querySelector('meta[name="browser-errors-url"]')?.content;if(t){if(e.error.stacktrace.some(e=>S.test(e.filename)||S.test(e.function))){d=!0;return}f++;try{await fetch(t,{method:"post",body:JSON.stringify(e)})}catch{}}}function v(e){return{type:e.name,value:e.message,stacktrace:w(e),catalogService:e.catalogService}}function b(e,t={}){return Object.assign({error:e,sanitizedUrl:`${window.location.protocol}//${window.location.host}${function(){let e=i("meta[name=analytics-location]");return e?e.content:window.location.pathname}()+function(){let e=i("meta[name=analytics-location-query-strip]"),t="";e||(t=window.location.search);let n=i("meta[name=analytics-location-params]");for(let e of(n&&(t+=(t?"&":"?")+n.content),document.querySelectorAll("meta[name=analytics-param-rename]"))){let n=e.content.split(":",2);t=t.replace(RegExp(`(^|[?&])${n[0]}($|=)`,"g"),`$1${n[1]}$2`)}return t}()}`||window.location.href,readyState:document.readyState,referrer:(0,u.wP)(),timeSinceLoad:Math.round(Date.now()-m),user:function(){let e=document.head?.querySelector('meta[name="user-login"]')?.content;if(e)return e;let t=(0,r.b)();return`anonymous-${t}`}()||void 0,bundler:s.A,ui:!!document.querySelector('meta[name="ui"]')},t)}function w(e){return(0,a.Q)(e.stack||"").map(e=>({filename:e.file||"",function:String(e.methodName),lineno:(e.lineNumber||0).toString(),colno:(e.column||0).toString()}))}let S=/(chrome|moz|safari)-extension:\/\//,_=!1;if(l.iG?.addEventListener("pageshow",()=>_=!1),l.iG?.addEventListener("pagehide",()=>_=!0),"function"==typeof BroadcastChannel){let e=new BroadcastChannel("shared-worker-error");e.addEventListener("message",e=>{y(e.data.error)})}let E=["Object Not Found Matching Id","Not implemented on this platform","provider because it's not your default extension"]},18515:(e,t,n)=>{n.d(t,{$:()=>c,c:()=>l});var r=n(15205),o=n(40578),a=n(71207);function i(){let e=(0,o.dU)().featureFlags,t=e.map(e=>e.toUpperCase());return new Set(t)}let s=a.W6?i:(0,r.Z)(i);function l(e){return s().has(e.toUpperCase())}let c={isFeatureEnabled:l}},80702:(e,t,n)=>{n.d(t,{Bt:()=>a,DN:()=>s,KL:()=>u,Se:()=>i,qC:()=>d,sw:()=>l});var r=n(31347);function o(e,t,n){return e.dispatchEvent(new CustomEvent(t,{bubbles:!0,cancelable:n}))}function a(e,t){t&&(function(e,t){if(!(e instanceof HTMLFormElement))throw TypeError("The specified element is not of type HTMLFormElement.");if(!(t instanceof HTMLElement))throw TypeError("The specified element is not of type HTMLElement.");if("submit"!==t.type)throw TypeError("The specified element is not a submit button.");if(!e||e!==t.form)throw Error("The specified element is not owned by the form element.")}(e,t),(0,r.j)(t)),o(e,"submit",!0)&&e.submit()}function i(e,t){if("boolean"==typeof t){if(e instanceof HTMLInputElement)e.checked=t;else throw TypeError("only checkboxes can be set to boolean value")}else{if("checkbox"===e.type)throw TypeError("checkbox can't be set to string value");e.value=t}o(e,"change",!1)}function s(e,t){for(let n in t){let r=t[n],o=e.elements.namedItem(n);o instanceof HTMLInputElement?o.value=r:o instanceof HTMLTextAreaElement&&(o.value=r)}}function l(e){if(!(e instanceof HTMLElement))return!1;let t=e.nodeName.toLowerCase(),n=(e.getAttribute("type")||"").toLowerCase();return"select"===t||"textarea"===t||"input"===t&&"submit"!==n&&"reset"!==n||e.isContentEditable}function c(e){return new URLSearchParams(e)}function u(e,t){let n=new URLSearchParams(e.search),r=c(t);for(let[e,t]of r)n.append(e,t);return n.toString()}function d(e){return c(new FormData(e)).toString()}},31347:(e,t,n)=>{function r(e){let t=e.closest("form");if(!(t instanceof HTMLFormElement))return;let n=o(t);if(e.name){let r=e.matches("input[type=submit]")?"Submit":"",o=e.value||r;n||((n=document.createElement("input")).type="hidden",n.classList.add("js-submit-button-value"),t.prepend(n)),n.name=e.name,n.value=o}else n&&n.remove()}function o(e){let t=e.querySelector("input.js-submit-button-value");return t instanceof HTMLInputElement?t:null}n.d(t,{j:()=>r,u:()=>o})},79697:(e,t,n)=>{let r;n.d(t,{YT:()=>f,qP:()=>m,yM:()=>p});var o=n(88149),a=n(86058),i=n(56363),s=n(43786);let{getItem:l}=(0,i.Z)("localStorage"),c="dimension_",u=["utm_source","utm_medium","utm_campaign","utm_term","utm_content","scid"];try{let e=(0,o.n)("octolytics");delete e.baseContext,r=new a.R(e)}catch(e){}function d(e){let t=(0,o.n)("octolytics").baseContext||{};if(t)for(let[e,n]of(delete t.app_id,delete t.event_url,delete t.host,Object.entries(t)))e.startsWith(c)&&(t[e.replace(c,"")]=n,delete t[e]);let n=document.querySelector("meta[name=visitor-payload]");if(n){let e=JSON.parse(atob(n.content));Object.assign(t,e)}let r=new URLSearchParams(window.location.search);for(let[e,n]of r)u.includes(e.toLowerCase())&&(t[e]=n);return t.staff=(0,s.B)().toString(),Object.assign(t,e)}function f(e){r?.sendPageView(d(e))}function m(e,t={}){let n=document.head?.querySelector('meta[name="current-catalog-service"]')?.content,o=n?{service:n}:{};for(let[e,n]of Object.entries(t))null!=n&&(o[e]=`${n}`);if(r){let t=e||"unknown";d(o),r.sendEvent(t,d(o))}}function p(e){return Object.fromEntries(Object.entries(e).map(([e,t])=>[e,JSON.stringify(t)]))}},5223:(e,t,n)=>{n.d(t,{r:()=>s});var r=n(47940),o=n(50655);let a="parse-html-no-op",i=r.ZO.createPolicy(a,{createHTML:e=>o.O.apply({policy:()=>e,policyName:a,fallback:e,sanitize:!1,fallbackOnError:!0})});function s(e,t){let n=e.createElement("template");return n.innerHTML=i.createHTML(t),e.importNode(n.content,!0)}},54508:(e,t,n)=>{n.d(t,{A:()=>s,k:()=>i});var r=n(15205),o=n(71207);let a=(0,r.Z)(function(){return o.n4?.head?.querySelector('meta[name="runtime-environment"]')?.content||""}),i=(0,r.Z)(function(){return"enterprise"===a()}),s="webpack"},56363:(e,t,n)=>{n.d(t,{Z:()=>i,_:()=>s});var r=n(71207),o=n(43786);let a=class NoOpStorage{getItem(){return null}setItem(){}removeItem(){}clear(){}key(){return null}get length(){return 0}};function i(e,t={throwQuotaErrorsOnSet:!1},n=r.iG,i=e=>e,s=e=>e){let l;try{if(!n)throw Error();l=n[e]||new a}catch{l=new a}let{throwQuotaErrorsOnSet:c}=t;function u(e){t.sendCacheStats&&(0,o.b)({incrementKey:e})}function d(e){try{if(l.removeItem(e),t.ttl){let t=`${e}:expiry`;l.removeItem(t)}}catch(e){}}return{getItem:function(e,t=new Date().getTime()){try{let n=l.getItem(e);if(!n)return null;let r=`${e}:expiry`,o=Number(l.getItem(r));if(o&&t>o)return d(e),d(r),u("SAFE_STORAGE_VALUE_EXPIRED"),null;return u("SAFE_STORAGE_VALUE_WITHIN_TTL"),i(n)}catch(e){return null}},setItem:function(e,n,r=new Date().getTime()){try{if(l.setItem(e,s(n)),t.ttl){let n=`${e}:expiry`,o=r+t.ttl;l.setItem(n,o.toString())}}catch(e){if(c&&e instanceof Error&&e.message.toLowerCase().includes("quota"))throw e}},removeItem:d,clear:l.clear,key:l.key,get length(){return l.length}}}function s(e){return i(e,{throwQuotaErrorsOnSet:!1},r.iG,JSON.parse,JSON.stringify)}},65706:(e,t,n)=>{n.d(t,{LS:()=>a,cl:()=>i,rV:()=>o});var r=n(56363);let{getItem:o,setItem:a,removeItem:i}=(0,r.Z)("sessionStorage")},58374:(e,t,n)=>{n.d(t,{Ak:()=>v,CI:()=>_,Gj:()=>h,J$:()=>S,Nb:()=>E,Nt:()=>b,OE:()=>y,U6:()=>f,Wl:()=>g,XL:()=>d,jN:()=>o,pS:()=>m,sj:()=>p,vu:()=>T,wG:()=>L,wP:()=>w});var r=n(65706);let o="reload",a="soft-nav:fail",i="soft-nav:fail-referrer",s="soft-nav:referrer",l="soft-nav:marker",c="soft-nav:react-app-name",u="soft-nav:latest-mechanism";function d(){(0,r.LS)(l,"0"),(0,r.cl)(s),(0,r.cl)(a),(0,r.cl)(i),(0,r.cl)(c),(0,r.cl)(u)}function f(e){(0,r.LS)(l,e)}function m(){(0,r.LS)(l,"0")}function p(){let e=(0,r.rV)(l);return e&&"0"!==e}function h(){return(0,r.rV)(l)}function y(){return!!g()}function g(){return(0,r.rV)(a)}function v(e){(0,r.LS)(a,e||o),(0,r.LS)(i,window.location.href)}function b(){(0,r.LS)(s,window.location.href)}function w(){return(0,r.rV)(s)||document.referrer}function S(){let e=E();e?(0,r.LS)(c,e):(0,r.cl)(c)}function _(){return(0,r.rV)(c)}function E(){return document.querySelector('meta[name="ui"]')?"ui":document.querySelector("react-app")?.getAttribute("app-name")}function T(e){(0,r.LS)(u,e)}function L(){return(0,r.rV)(u)}},71207:(e,t,n)=>{n.d(t,{Qg:()=>r.Qg,W6:()=>r.W6,cF:()=>r.cF,iG:()=>o.iG,n4:()=>o.n4,ssrSafeLocation:()=>o.jX,zu:()=>o.zu});var r=n(60327),o=n(66199)},66199:(e,t,n)=>{n.d(t,{iG:()=>o,jX:()=>i,n4:()=>r,zu:()=>a});let r="undefined"==typeof document?void 0:document,o="undefined"==typeof window?void 0:window,a="undefined"==typeof history?void 0:history,i="undefined"==typeof location?{pathname:"",origin:"",search:"",hash:""}:location},60327:(e,t,n)=>{n.d(t,{Qg:()=>a,W6:()=>o,cF:()=>i});var r=n(66199);let o=void 0===r.n4,a=!o;function i(){return!!o||!!r.n4.querySelector('react-app[data-ssr="true"]')}},43786:(e,t,n)=>{n.d(t,{B:()=>f,b:()=>l});var r=n(71207),o=n(87274),a=n(54508),i=n(18515);let s=[];function l(e,t=!1,n=1){if(!0!==(0,i.c)("BROWSER_STATS_DISABLED")){if(n<0||n>1)throw RangeError("Sampling probability must be between 0 and 1");void 0===e.timestamp&&(e.timestamp=new Date().getTime()),e.loggedIn=!!r.n4?.head?.querySelector('meta[name="user-login"]')?.content,e.staff=f(),e.bundler=a.A,Math.random()<n&&s.push(e),t?d():u()}}let c=null;async function u(){await o.C,null==c&&(c=window.requestIdleCallback(d))}function d(){if(c=null,!s.length)return;let e=r.n4?.head?.querySelector('meta[name="browser-stats-url"]')?.content;if(!e)return;let t=function(e){let t=[],n=e.map(e=>JSON.stringify(e));for(;n.length>0;)t.push(function(e){let t=e.shift(),n=[t],r=t.length;for(;e.length>0&&r<=65536;){let t=e[0].length;if(r+t<=65536){let o=e.shift();n.push(o),r+=t}else break}return n}(n));return t}(s);for(let n of t)!function(e,t){try{navigator.sendBeacon&&navigator.sendBeacon(e,t)}catch{}}(e,`{"stats": [${n.join(",")}] }`);s=[]}function f(){return!!r.n4?.head?.querySelector('meta[name="user-staff"]')?.content}r.n4?.addEventListener("pagehide",d),r.n4?.addEventListener("visibilitychange",d)},50655:(e,t,n)=>{n.d(t,{O:()=>c,d:()=>TrustedTypesPolicyError});var r=n(18515),o=n(43786),a=n(65809),i=n(27856),s=n.n(i),l=n(79697);let TrustedTypesPolicyError=class TrustedTypesPolicyError extends Error{};let c={apply:function({policy:e,policyName:t,fallback:n,fallbackOnError:i=!1,sanitize:c,silenceErrorReporting:u=!1}){try{if((0,r.c)("BYPASS_TRUSTED_TYPES_POLICY_RULES"))return n;(0,o.b)({incrementKey:"TRUSTED_TYPES_POLICY_CALLED",trustedTypesPolicyName:t},!1,.1);let a=e();return c&&new Promise(e=>{let n=window.performance.now(),r=s().sanitize(a,{FORBID_ATTR:[]}),o=window.performance.now();if(a.length!==r.length){let i=Error("Trusted Types policy output sanitized"),s=i.stack?.slice(0,1e3),c=a.slice(0,250);(0,l.qP)("trusted_types_policy.sanitize",{policyName:t,output:c,stack:s,outputLength:a.length,sanitizedLength:r.length,executionTime:o-n}),e(a)}}),a}catch(e){if(e instanceof TrustedTypesPolicyError||(u||(0,a.eK)(e),(0,o.b)({incrementKey:"TRUSTED_TYPES_POLICY_ERROR",trustedTypesPolicyName:t}),!i))throw e}return n}}},47940:(e,t,n)=>{n.d(t,{ZO:()=>u});var r,o=n(71207),a=n(43786);function i(e){return()=>{throw TypeError(`The policy does not implement the function ${e}`)}}let s={createHTML:i("createHTML"),createScript:i("createScript"),createScriptURL:i("createScriptURL")},l=(r=globalThis).__TRUSTED_TYPE_POLICIES__??(r.__TRUSTED_TYPE_POLICIES__=new Map),c=globalThis.trustedTypes??{createPolicy:(e,t)=>({name:e,...s,...t})},u={createPolicy:(e,t)=>{if(l.has(e))return(0,a.b)({incrementKey:"TRUSTED_TYPES_POLICY_INITIALIZED_TWICE"}),l.get(e);{let n=Object.freeze(c.createPolicy(e,t));return l.set(e,n),n}}},d=!1;o.n4?.addEventListener("securitypolicyviolation",e=>{"require-trusted-types-for"!==e.violatedDirective||d||(console.warn(`Hi fellow Hubber!
    You're probably seeing a Report Only Trusted Types error near this message. This is intended behaviour, staff-only,
    does not impact application control flow, and is used solely for statistic collection. Unfortunately we
    can't gather these statistics without adding the above warnings to your console. Sorry about that!
    Feel free to drop by #pse-architecture if you have any additional questions about Trusted Types or CSP.`),d=!0)})}},e=>{var t=t=>e(e.s=t);e.O(0,["vendors-node_modules_dompurify_dist_purify_js","vendors-node_modules_github_selector-observer_dist_index_esm_js","vendors-node_modules_morphdom_dist_morphdom-esm_js-node_modules_github_memoize_dist_esm_index_js","vendors-node_modules_virtualized-list_es_index_js-node_modules_github_template-parts_lib_index_js","vendors-node_modules_github_remote-form_dist_index_js-node_modules_delegated-events_dist_inde-e53a3f","app_assets_modules_github_ref-selector_ts"],()=>t(61013)),e.O()}]);
//# sourceMappingURL=codespaces-456c5f7b254c.js.map