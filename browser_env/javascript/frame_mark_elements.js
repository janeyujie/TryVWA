/**
 * Go through all DOM elements in the frame (including shadowDOMs), give them unique browsergym
 * identifiers (bid), and store custom data in the aria-roledescription attribute.
 *
 * MODIFIED VERSION: This script now includes a MutationObserver to dynamically
 * mark new elements that are added to the DOM after the initial page load.
 */
var { innerWidth: windowWidth, innerHeight: windowHeight } = window;
var scrollX = window.scrollX || document.documentElement.scrollLeft;
var scrollY = window.scrollY || document.documentElement.scrollTop;

([parent_bid, bid_attr_name, iframe_position, super_iframe_offset]) => {

    // standard html tags
    const html_tags = [
        "a", "abbr", "acronym", "address", "applet", "area", "article", "aside", "audio",
        "b", "base", "basefont", "bdi", "bdo", "big", "blockquote", "body", "br", "button",
        "canvas", "caption", "center", "cite", "code", "col", "colgroup", "data", "datalist",
        "dd", "del", "details", "dfn", "dialog", "dir", "div", "dl", "dt", "em", "embed",
        "fieldset", "figcaption", "figure", "font", "footer", "form", "frame", "frameset",
        "h1", "h2", "h3", "h4", "h5", "h6", "head", "header", "hgroup", "hr", "html", "i",
        "iframe", "img", "input", "ins", "kbd", "label", "legend", "li", "link", "main",
        "map", "mark", "menu", "meta", "meter", "nav", "noframes", "noscript", "object",
        "ol", "optgroup", "option", "output", "p", "param", "picture", "pre", "progress",
        "q", "rp", "rt", "ruby", "s", "samp", "script", "search", "section", "select",
        "small", "source", "span", "strike", "strong", "style", "sub", "summary", "sup",
        "svg", "table", "tbody", "td", "template", "textarea", "tfoot", "th", "thead",
        "time", "title", "tr", "track", "tt", "u", "ul", "var", "video", "wbr"
    ];

    if (super_iframe_offset == null) {
        iframe_offset = { x: scrollX, y: scrollY, right: windowWidth, bottom: windowHeight };
    }
    else {
        [super_x, super_y, super_right, super_bottom] = [super_iframe_offset["x"], super_iframe_offset["y"], super_iframe_offset["right"], super_iframe_offset["bottom"]];

        x = Math.max(-iframe_position.x, 0);
        y = Math.max(-iframe_position.y, 0);
        right = Math.min(...[super_right, windowWidth,  super_right - iframe_position.x]);
        bottom = Math.min(...[super_bottom, windowHeight, super_bottom - iframe_position.y]);
        iframe_offset = { x: x, y: y, right: right, bottom: bottom };
    }

    let browsergym_first_visit = false;
    // if not yet set, set the frame (local) element counter to 0
    if (!("browsergym_frame_elem_counter" in window)) {
        window.browsergym_frame_elem_counter = 0;
        browsergym_first_visit = true;
    }

    // ========================================================================
    // START: Encapsulate the marking logic into a reusable function
    // ========================================================================
    const markElements = (elementsToMark) => {
        let i = 0;
        while (i < elementsToMark.length) {
            const elem = elementsToMark[i];

            // Handle shadowDOMs
            if (elem.shadowRoot !== null) {
                const shadowElements = Array.from(elem.shadowRoot.querySelectorAll("*"));
                elementsToMark = new Array(
                    ...Array.prototype.slice.call(elementsToMark, 0, i + 1),
                    ...shadowElements,
                    ...Array.prototype.slice.call(elementsToMark, i + 1)
                );
                // Recursively mark elements in the shadow root
                markElements(shadowElements);
            }
            i++;

            // Skip non-standard HTML tags or elements that are already marked
            if (!elem.tagName || !html_tags.includes(elem.tagName.toLowerCase()) || elem.hasAttribute(bid_attr_name)) {
                continue;
            }

            // Write dynamic values to DOM attributes
            if (typeof elem.value !== 'undefined') {
                elem.setAttribute("value", elem.value);
            }
            if (typeof elem.checked !== 'undefined') {
                elem.setAttribute(elem.checked ? "checked" : "no-checked", ""); // Avoid removing/adding attribute, which can trigger observer
            }

            // Assign a new element ID
            let elem_local_id = window.browsergym_frame_elem_counter++;
            let elem_global_bid = parent_bid === "" ? `${elem_local_id}` : `${parent_bid}-${elem_local_id}`;
            elem.setAttribute(bid_attr_name, elem_global_bid);

            // Get position info and set custom attributes
            let [rect, is_in_viewport] = getElementPositionInfo(elem, iframe_offset, iframe_position);
            let left = (rect.left + iframe_position.x).toString();
            let top = (rect.top + iframe_position.y).toString();
            let right = (rect.right + iframe_position.x).toString();
            let bottom = (rect.bottom + iframe_position.y).toString();
            let center_x = ((rect.left + rect.right) / 2 + iframe_position.x).toString();
            let center_y = ((rect.top + rect.bottom) / 2 + iframe_position.y).toString();

            elem.setAttribute("browsergym_center", `(${center_x}, ${center_y})`);
            elem.setAttribute("browsergym_bounding_box", `(${left}, ${top}, ${right}, ${bottom})`);
            elem.setAttribute("browsergym_is_in_viewport", `${is_in_viewport}`);

            let original_content = elem.getAttribute("aria-roledescription") || "";
            let new_content = `${elem_global_bid}_${left}_${top}_${center_x}_${center_y}_${right}_${bottom}_${is_in_viewport}_${original_content}`;
            elem.setAttribute("aria-roledescription", new_content);
        }
    };
    // ========================================================================
    // END: Reusable marking function
    // ========================================================================


    // Initial marking of all elements on the page
    let allElements = Array.from(document.querySelectorAll('*'));
    markElements(allElements);

    // ========================================================================
    // START: Setup MutationObserver to mark dynamically added elements
    // ========================================================================
    // Only set up the observer on the first visit to avoid multiple observers
    if (browsergym_first_visit) {
        const observer = new MutationObserver((mutationsList, observer) => {
            for (const mutation of mutationsList) {
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                    mutation.addedNodes.forEach(node => {
                        // We only care about element nodes
                        if (node.nodeType === Node.ELEMENT_NODE) {
                            // Mark the new node itself and all of its descendants
                            let newElements = [node, ...Array.from(node.querySelectorAll('*'))];
                            markElements(newElements);
                        }
                    });
                }
            }
        });

        // Start observing the document body for added nodes
        observer.observe(document.body, { childList: true, subtree: true });

        // Store the observer on the window object so it can be disconnected later if needed
        window.browsergym_mutation_observer = observer;
    }
    // ========================================================================
    // END: MutationObserver setup
    // ========================================================================

    return iframe_offset;
};


// The helper functions below remain unchanged.
function getElementPositionInfo(element, iframe_offset, iframe_position) {
    var rect = element.getBoundingClientRect();
    let x = (rect.left + rect.right) / 2 ;
    let y = (rect.top + rect.bottom) / 2 ;
    //loop over element ancestors (parent) and refine iframe offset to be the most precise possible
    var parent = element.parentElement;
    parent_iframe_offset = { x: 0, y: 0, right: windowWidth, bottom: windowHeight };
    while (parent !== null) {
        var parent_rect = parent.getBoundingClientRect();
        parent_iframe_offset["x"] = Math.max(parent_rect.left , parent_iframe_offset["x"]  );
        parent_iframe_offset["y"] = Math.max(parent_rect.top , parent_iframe_offset["y"] );
        parent_iframe_offset["right"] = Math.min(parent_rect.right , parent_iframe_offset["right"]  );
        parent_iframe_offset["bottom"] = Math.min(parent_rect.bottom , parent_iframe_offset["bottom"] );
        parent = parent.parentElement;
    }

    var is_in_viewport = (
        x >= iframe_offset["x"] &&
        y >= iframe_offset["y"] &&
        x <= iframe_offset["right"] &&
        y <= iframe_offset["bottom"]
    );
    //this features is broken for the moment
    var NotBehindParent = (
        x >= parent_iframe_offset["x"] &&
        y >= parent_iframe_offset["y"] &&
        x <= parent_iframe_offset["right"] &&
        y <= parent_iframe_offset["bottom"]
    );

    var isVisible = (typeof element.offsetWidth === 'undefined' || typeof element.offsetHeight === 'undefined') || (element.offsetWidth > 0 && element.offsetHeight > 0);

    // Return true if the element is both in the viewport and has non-zero dimensions
    return [rect, (is_in_viewport  && isVisible && IsInFront(element))? 1 : 0];
}


function IsInFront(element){
    var rect = element.getBoundingClientRect();
    var x = (rect.left + rect.right) / 2 ;
    var y = (rect.top + rect.bottom) / 2 ;
    var newElement = elementFromPoint(x, y); //return the element in the foreground at position (x,y)
    if(newElement){
        if(newElement === element)
            return true;
    }
    return false;
}

function elementFromPoint(x, y) {
    let node = document.elementFromPoint(x, y);

    let child = node?.shadowRoot?.elementFromPoint(x, y);

    while (child && child !== node) {
      node = child;
      child = node?.shadowRoot?.elementFromPoint(x, y);
    }

    return child || node;
  }